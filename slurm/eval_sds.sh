#!/bin/bash
# Eval-only: produce outputs/sds_final/eval.json from the already-generated 10 final images.
#   Submit:  sbatch slurm/eval_sds.sh
#
#SBATCH --account=sagieb
#SBATCH --job-name=hw3_eval
#SBATCH --gres=gg:g0:1
#SBATCH --killable
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0:20:00
#SBATCH --output=/cs/labs/sagieb/yaara.berg/3D-HW3/slurm/logs/eval_%j.out
#SBATCH --error=/cs/labs/sagieb/yaara.berg/3D-HW3/slurm/logs/eval_%j.err

set -e
source /etc/profile.d/huji-lmod.sh
module load cuda/12.1 2>/dev/null || module load cuda/12.4.1 2>/dev/null || true

REPO_DIR=/cs/labs/sagieb/yaara.berg/3D-HW3
PYTHON=/cs/labs/sagieb/yaara.berg/miniconda3/envs/hw3/bin/python
cd "$REPO_DIR"

"$PYTHON" eval.py --fdir1 outputs/sds_final
echo "Done -> outputs/sds_final/eval.json"
