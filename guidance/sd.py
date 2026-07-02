from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] loading stable diffusion...')

        # stabilityai/stable-diffusion-2-1-base was deprecated/removed from the Hub;
        # Manojb/stable-diffusion-2-1-base is a community mirror with identical weights.
        model_key = "Manojb/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        # Keep a copy of the original attention processors for the frozen UNet path.
        self.base_attn_procs = dict(self.unet.attn_processors)
        self.lora_attn_procs = None
        if getattr(args, "loss_type", None) == "vsd":
            lora_rank = getattr(args, "lora_rank", 4)
            self._init_lora(lora_rank)
            print(f"[INFO] initialized LoRA (rank={lora_rank}) for VSD")

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings
    
    
    def _set_attn_processors(self, use_lora=False):
        if use_lora:
            if self.lora_attn_procs is None:
                raise RuntimeError("LoRA is not initialized. Use --loss_type vsd.")
            self.unet.set_attn_processor(self.lora_attn_procs)
        else:
            self.unet.set_attn_processor(self.base_attn_procs)

    def _init_lora(self, rank=4):
        """Attach trainable LoRA attention processors to the UNet (phi in VSD)."""
        lora_attn_procs = {}
        for name in self.base_attn_procs:
            cross_attention_dim = (
                None if "attn1" in name else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    int(name.split(".")[1])
                ]
            elif name.startswith("down_blocks"):
                hidden_size = self.unet.config.block_out_channels[int(name.split(".")[1])]
            else:
                raise ValueError(f"Unrecognized attention processor name: {name}")

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
            )

        self.lora_attn_procs = lora_attn_procs
        self.unet.set_attn_processor(self.lora_attn_procs)
        self.unet.requires_grad_(False)
        for proc in self.lora_attn_procs.values():
            for param in proc.parameters():
                param.requires_grad_(True)

    def get_lora_parameters(self):
        if self.lora_attn_procs is None:
            return []
        params = []
        for proc in self.lora_attn_procs.values():
            params.extend(proc.parameters())
        return params

    def sample_timestep(self, batch_size, device):
        """Sample a random diffusion timestep in the assignment t_range."""
        return torch.randint(
            self.min_step,
            self.max_step + 1,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100, use_lora=False):
        self._set_attn_processors(use_lora=use_lora)
        latent_model_input = torch.cat([latents_noisy] * 2)

        tt = torch.cat([t] * 2)
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        return noise_pred


    def get_sds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
    ):
        
        # TODO: Implement the loss function for SDS
        raise NotImplementedError("SDS is not implemented yet.")
    
    
    def compute_posterior_mean(self, xt, noise_pred, t, t_prev):
        """DDPM posterior mean mu(x^t, c, eps_theta) used in the reverse step.

        One reverse step is x^{t-1} = mu + sigma^t * z^t. Here mu is a linear
        combination of the one-step denoised estimate x0_hat and the current
        noisy latent x^t (see PDS paper Eq. 7-8).

        Args:
            xt: Noisy latent x^t at timestep t, shape [B, C, H, W].
            noise_pred: CFG noise prediction eps_theta(x^t, c, t).
            t: Current diffusion timestep indices.
            t_prev: Previous timestep indices (t - 1).

        Returns:
            Posterior mean mu with the same shape as xt.
        """
        betas = self.scheduler.betas.to(self.device)
        alphas = self.scheduler.alphas.to(self.device)
        alpha_bar_t = self.alphas[t]
        alpha_bar_t_prev = self.alphas[t_prev]
        beta_t = betas[t]

        # x0_hat = (x^t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
        pred_x0 = (xt - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        # mu = c0 * x0_hat + c1 * x^t  (DDPM posterior mean coefficients)
        c0 = alpha_bar_t_prev.sqrt() * beta_t / (1 - alpha_bar_t)
        c1 = alphas[t].sqrt() * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        return c0 * pred_x0 + c1 * xt

    def compute_stochastic_latent(
        self, x0, text_embeddings, noise, noise_prev, t, t_prev, guidance_scale,
    ):
        """Extract the stochastic latent z_tilde^t from a clean latent x^0.

        Rearranges the reverse DDPM step x^{t-1} = mu + sigma^t * z^t into:
            z_tilde^t = (x^{t-1} - mu) / sigma^t

        Source and target must use the same `noise` (eps^t) and `noise_prev`
        (eps^{t-1}) so that differences reflect prompts/latents, not sampling.

        Args:
            x0: Clean latent x^0, shape [B, C, H, W].
            text_embeddings: Stacked [uncond, cond] embeddings for CFG.
            noise: Shared forward-noise eps^t for building x^t.
            noise_prev: Shared forward-noise eps^{t-1} for building x^{t-1}.
            t: Current timestep indices.
            t_prev: Previous timestep indices (t - 1).
            guidance_scale: Classifier-free guidance weight.

        Returns:
            Stochastic latent z_tilde^t with the same shape as x0.
        """
        # x^t = sqrt(alpha_bar_t) * x^0 + sqrt(1 - alpha_bar_t) * eps^t
        latents_noisy = self.scheduler.add_noise(x0, noise, t)
        noise_pred = self.get_noise_preds(latents_noisy, t, text_embeddings, guidance_scale)
        # x^{t-1} built with the same eps^{t-1} for both source and target
        x_t_prev = self.scheduler.add_noise(x0, noise_prev, t_prev)

        betas = self.scheduler.betas.to(self.device)
        alpha_bar_t = self.alphas[t]
        alpha_bar_t_prev = self.alphas[t_prev]
        # sigma^t = sqrt((1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t)
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * betas[t]).sqrt()

        mu = self.compute_posterior_mean(latents_noisy, noise_pred, t, t_prev)
        # Broadcast sigma^t over spatial dims for batched timesteps
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        return (x_t_prev - mu) / sigma_t

    def pds_timestep_sampling(self, batch_size, device):
        """Sample (t, t_prev) pairs from the DDIM inference schedule.

        The schedule is reversed so t_prev < t in noise level (one denoising step back).
        """
        self.scheduler.set_timesteps(self.num_train_timesteps)
        # Ascending noise level: ..., t_prev, t, ... toward cleaner latents
        timesteps = torch.flip(self.scheduler.timesteps, [0]).to(device)

        min_idx = max(1, int(len(timesteps) * self.t_range[0]))
        max_idx = max(min_idx + 1, int(len(timesteps) * self.t_range[1]))
        idx = torch.randint(min_idx, max_idx, (batch_size,), device=device, dtype=torch.long)

        t = timesteps[idx]
        t_prev = timesteps[idx - 1]
        return t, t_prev

    def get_pds_loss(
        self, src_latents, tgt_latents, 
        src_text_embedding, tgt_text_embedding,
        guidance_scale=7.5, 
        grad_scale=1,
    ):
        """Posterior Distillation Sampling (PDS) loss for text-guided editing.

        Edits tgt_latents toward edit_prompt while preserving source structure by
        matching stochastic latents:

            grad L_pds / d(x_tgt^0) ~ E_{t, eps^t, eps^{t-1}} [z_tilde_tgt - z_tilde_src]

        The loss is implemented via the SDS reparameterization trick:
            L = 0.5 * ||x_tgt^0 - (x_tgt^0 - grad).detach()||^2,  grad = z_tilde_tgt - z_tilde_src
        so backprop flows only through x_tgt^0 (not the frozen UNet).

        Args:
            src_latents: VAE-encoded source image latent (fixed during optimization).
            tgt_latents: Target latent being optimized, initialized from source.
            src_text_embedding: [uncond, src_cond] embeddings for the source prompt.
            tgt_text_embedding: [uncond, tgt_cond] embeddings for the edit prompt.
            guidance_scale: CFG weight (7.5 for PDS per assignment).
            grad_scale: Optional scalar on the distilled gradient.

        Returns:
            Scalar PDS loss tensor with gradients w.r.t. tgt_latents.
        """
        batch_size = tgt_latents.shape[0]
        t, t_prev = self.pds_timestep_sampling(batch_size, tgt_latents.device)

        # Shared noise variables — required so src/tgt differ only in x^0 and prompt
        noise = torch.randn_like(tgt_latents)
        noise_prev = torch.randn_like(tgt_latents)

        z_src = self.compute_stochastic_latent(
            src_latents.detach(), src_text_embedding,
            noise, noise_prev, t, t_prev, guidance_scale,
        )
        z_tgt = self.compute_stochastic_latent(
            tgt_latents, tgt_text_embedding,
            noise, noise_prev, t, t_prev, guidance_scale,
        )

        # grad = z_tgt - z_src so the MSE trick moves tgt toward src's trajectory
        grad = grad_scale * (z_tgt - z_src)
        grad = torch.nan_to_num(grad)
        # Detach target so UNet weights are not updated; grad still flows via MSE trick
        target = (tgt_latents - grad).detach()
        loss = 0.5 * F.mse_loss(tgt_latents, target, reduction="mean")
        return loss

    def get_lora_train_loss(
        self,
        latents,
        text_embeddings,
        guidance_scale=7.5,
    ):
        """Phase A: train LoRA phi to predict noise on the current latent x^0.

        Standard diffusion objective on noisy latents built from the frozen x^0:
            L_phi = ||epsilon - epsilon_phi(x^t, c, t)||^2
        """
        batch_size = latents.shape[0]
        t = self.sample_timestep(batch_size, latents.device)
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents.detach(), noise, t)
        noise_pred_phi = self.get_noise_preds(
            latents_noisy, t, text_embeddings, guidance_scale, use_lora=True,
        )
        return F.mse_loss(noise_pred_phi, noise, reduction="mean")

    def get_vsd_loss(
        self,
        latents,
        text_embeddings,
        guidance_scale=7.5,
        grad_scale=1,
    ):
        """Phase B: Variational Score Distillation loss for updating x^0.

        grad L_vsd / d(x^0) ~ E_{t, epsilon} [epsilon_theta(x^t, c, t) - epsilon_phi(x^t, c, t)]

        LoRA phi is treated as fixed during this step (epsilon_phi is detached).
        """
        batch_size = latents.shape[0]
        t = self.sample_timestep(batch_size, latents.device)
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        with torch.no_grad():
            noise_pred_theta = self.get_noise_preds(
                latents_noisy, t, text_embeddings, guidance_scale, use_lora=False,
            )
            noise_pred_phi = self.get_noise_preds(
                latents_noisy, t, text_embeddings, guidance_scale, use_lora=True,
            )

        grad = grad_scale * (noise_pred_theta - noise_pred_phi)
        grad = torch.nan_to_num(grad)
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="mean")
        return loss

    @torch.no_grad()
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
