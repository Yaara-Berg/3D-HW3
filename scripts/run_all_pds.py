#!/usr/bin/env python3
"""Run PDS editing for every pair in prompt_img_pairs.json, then evaluate."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_path(path: str, repo_root: Path) -> str:
    path = os.path.expandvars(path).replace("$HOME", str(Path.home()))
    p = Path(path)
    if not p.is_absolute():
        p = repo_root / p
    return str(p.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs_json",
        type=str,
        default="data/prompt_img_pairs.json",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="outputs/pds",
        help="Eval-ready folder: final edit_prompt.png files + eval.json",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="outputs/pds_work",
        help="Per-run scratch (run configs, optional intermediates)",
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument(
        "--clean_eval_dir",
        action="store_true",
        help="Remove existing PNG/JSON files in eval_dir before running.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Only run edits; do not run eval.py at the end.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pairs_path = resolve_path(args.pairs_json, repo_root)
    eval_dir = Path(resolve_path(args.eval_dir, repo_root))
    work_dir = Path(resolve_path(args.work_dir, repo_root))
    main_py = repo_root / "main.py"
    eval_py = repo_root / "eval.py"

    eval_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_eval_dir:
        for pattern in ("*.png", "eval.json"):
            for path in eval_dir.glob(pattern):
                path.unlink()

    with open(pairs_path) as f:
        pairs = json.load(f)

    for key, entry in pairs.items():
        prompt = entry["prompt"]
        edit_prompt = entry["edit_prompt"]
        src_img_path = resolve_path(entry["img_path"], repo_root)
        run_work_dir = work_dir / key
        run_work_dir.mkdir(parents=True, exist_ok=True)

        # main.py appends loss_type ("pds") to save_dir
        run_output_dir = run_work_dir / "pds"
        final_name = f"{edit_prompt.replace(' ', '_')}.png"
        final_src = run_output_dir / final_name
        final_dst = eval_dir / final_name

        cmd = [
            sys.executable,
            str(main_py),
            "--prompt", prompt,
            "--edit_prompt", edit_prompt,
            "--src_img_path", src_img_path,
            "--loss_type", "pds",
            "--guidance_scale", str(args.guidance_scale),
            "--step", str(args.step),
            "--device", str(args.device),
            "--precision", args.precision,
            "--save_dir", str(run_work_dir),
        ]
        print(f"\n[*] Running PDS for {key}")
        print("    ", " ".join(cmd))
        subprocess.run(cmd, cwd=repo_root, check=True)

        if not final_src.is_file():
            raise FileNotFoundError(f"Expected output not found: {final_src}")
        shutil.copy2(final_src, final_dst)
        print(f"[*] Copied eval image -> {final_dst}")

    if args.skip_eval:
        print(f"\n[*] Skipping eval. Final images are in {eval_dir}")
        return

    print(f"\n[*] Running CLIP eval on {eval_dir}")
    subprocess.run(
        [sys.executable, str(eval_py), "--fdir1", str(eval_dir)],
        cwd=repo_root,
        check=True,
    )
    print(f"\n[*] Done. Submission-ready outputs: {eval_dir}")


if __name__ == "__main__":
    main()
