#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <mode> <arch>"
  echo "  mode: adv | norm"
  echo "  arch: cnn | wrn28x10"
  exit 1
fi

mode="$1"
arch="$2"
name="${arch}_${mode}_meansparse"
# SEEDS=(0 42 123 9384 19302)
SEEDS=(42)

last=$(( ${#SEEDS[@]} - 1 ))

sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=20G
#SBATCH --account=m25146
#SBATCH --array=0-${last}
#SBATCH --job-name=${name}
#SBATCH --output=logs_meansparse/%x_%A_%a.out
#SBATCH --error=logs_meansparse/%x_%A_%a.err

set -euo pipefail

cd "\$SLURM_SUBMIT_DIR"
source venv/bin/activate

SEEDS=(42)
SEED=\${SEEDS[\$SLURM_ARRAY_TASK_ID]}

# MODEL="models/${arch}_${mode}_trained_seed_\${SEED}.pth"
MODEL="models/default_model.pth"
echo "Running MeanSparse for seed=\$SEED model=\$MODEL"

python meansparse_postprocess.py \
    --model-file "models/default_model.pth"\
    --arch "${arch}" \
    --alphas 0.001 0.01 0.05 0.1 0.15 0.19 0.2 0.25 0.3 0.32 0.33  0.35 

EOT
