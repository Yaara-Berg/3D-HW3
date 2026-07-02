#!/bin/bash
# Task 1 (SDS): generate one image per prompt in data/prompt_img_pairs.json, then eval CLIP score.
#   Submit:  sbatch slurm/sds.sh
#   Output:  outputs/sds/<prompt_with_underscores>.png  +  outputs/sds/eval.json
#
#SBATCH --account=sagieb
#SBATCH --job-name=hw3_sds
#SBATCH --gres=gg:g0:1            # cheap/killable GPU is plenty (SD-2-1 @ 64x64, 500 steps)
#SBATCH --killable
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/cs/labs/sagieb/yaara.berg/3D-HW3/slurm/logs/sds_%j.out
#SBATCH --error=/cs/labs/sagieb/yaara.berg/3D-HW3/slurm/logs/sds_%j.err

set -e

# CUDA (module system is not available on the login node — load it on the compute node).
source /etc/profile.d/huji-lmod.sh
module load cuda/12.1 2>/dev/null || module load cuda/12.4.1 2>/dev/null || true

REPO_DIR=/cs/labs/sagieb/yaara.berg/3D-HW3
PYTHON=/cs/labs/sagieb/yaara.berg/miniconda3/envs/hw3/bin/python
cd "$REPO_DIR"

nvidia-smi -L 2>/dev/null | head -4 || echo "(no nvidia-smi; continuing)"

GUIDANCE=25

# 10 prompts from data/prompt_img_pairs.json (the "prompt" field).
PROMPTS=(
  "A red bus driving on a desert road"
  "a boat in a river"
  "A cabin surrounded by forests"
  "A church beside a lake"
  "A villa close to the pool"
  "A castle next to a river"
  "A burger on the table"
  "A dog sitting on grass"
  "a cat sitting on a table"
  "A car on the road"
)

for p in "${PROMPTS[@]}"; do
  echo "==== SDS: $p ===="
  "$PYTHON" main.py --prompt "$p" --loss_type sds --guidance_scale "$GUIDANCE"
done

# Curate a clean submission folder with ONLY the 10 final prompt-named images.
# main.py also writes intermediate snapshots (0.png, 25.png, ...) into outputs/sds;
# eval.py globs every *.png, so those snapshots must be excluded or they tank the score.
echo "==== Building clean eval/submission folder outputs/sds_final ===="
rm -rf outputs/sds_final && mkdir -p outputs/sds_final
for f in outputs/sds/*.png; do
  b=$(basename "$f" .png)
  [[ "$b" =~ ^[0-9]+$ ]] || cp "$f" outputs/sds_final/"$b".png   # skip numeric snapshot files
done

echo "==== CLIP evaluation (on the 10 final images only) ===="
"$PYTHON" eval.py --fdir1 outputs/sds_final

echo "Done. Submit outputs/sds_final/ (10 images + eval.json)."
