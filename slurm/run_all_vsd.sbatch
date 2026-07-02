#!/bin/bash
#SBATCH --job-name=vsd_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --killable
#SBATCH --output=logs/vsd_all_%j.out
#SBATCH --error=logs/vsd_all_%j.err

set -euo pipefail

cd /cs/labs/dina/hadar_hever/3D-HW3
source hw3_env/bin/activate
mkdir -p logs outputs/vsd outputs/vsd_work

python - << "PY"
import json, subprocess, sys
from pathlib import Path

repo = Path(".")
pairs = json.load(open("data/prompt_img_pairs.json"))

for key, e in pairs.items():
    prompt = e["prompt"]
    run_dir = repo / "outputs" / "vsd_work" / key
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "main.py",
        "--prompt", prompt,
        "--loss_type", "vsd",
        "--guidance_scale", "7.5",
        "--step", "500",
        "--log_step", "25",
        "--device", "0",
        "--save_dir", str(run_dir),
    ]
    print("[*] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    src = run_dir / "vsd" / (prompt.replace(" ", "_") + ".png")
    dst = repo / "outputs" / "vsd" / (prompt.replace(" ", "_") + ".png")
    dst.write_bytes(src.read_bytes())
    print("[*] Copied:", dst)

print("[*] Done generating all VSD images")
PY

python eval.py --fdir1 outputs/vsd
echo "[*] Eval written to outputs/vsd/eval.json"
