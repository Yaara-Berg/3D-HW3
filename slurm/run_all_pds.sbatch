#!/bin/bash
#SBATCH --job-name=hw3_pds
#SBATCH --time=24:00:00
#SBATCH --gres=gg:g4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/cs/labs/dina/hadar_hever/3D-HW3/logs/pds_all_%j.log
#SBATCH --error=/cs/labs/dina/hadar_hever/3D-HW3/logs/pds_all_%j.err

set -euo pipefail

REPO_ROOT="/cs/labs/dina/hadar_hever/3D-HW3"
VENV="${REPO_ROOT}/hw3_env"

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

# shellcheck source=/dev/null
source "${VENV}/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/run_all_pds.py \
  --pairs_json data/prompt_img_pairs.json \
  --eval_dir outputs/pds \
  --work_dir outputs/pds_work \
  --guidance_scale 7.5 \
  --step 500 \
  --device 0 \
  --precision fp32 \
  --clean_eval_dir
