#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <mode> <arch> <activation>"
  echo "  mode: adv | norm"
  echo "  arch: cnn | wrn28x10 | resnet18"
  echo "  activation: relu | gelu"
  exit 1
fi


mode="$1"
arch="$2"
activation="$3"
name="${arch}_${mode}_${activation}_training"
outdir="logs"
mkdir -p "${outdir}"

# Choose adv flag depending on mode
if [ "$mode" = "adv" ]; then
  ADV_FLAG="--adv-train"
else
  ADV_FLAG=""
fi

# Define seeds
SEEDS=(42)
num_seeds=${#SEEDS[@]}
last_index=$((num_seeds - 1))

    sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=20G
#SBATCH --account=m25146
#SBATCH --job-name=${name}
#SBATCH --output=${outdir}/%x_%A_%a.out
#SBATCH --error=${outdir}/%x_%A_%a.err
#SBATCH --array=0-${last_index}

set -euo pipefail

cd "\$SLURM_SUBMIT_DIR"
source venv/bin/activate
export PYTHONPATH="\$SLURM_SUBMIT_DIR:\${PYTHONPATH-}"
export MPLBACKEND=Agg

log() { echo -e "[\$(date '+%Y-%m-%d %H:%M:%S')] \$*"; }

# Same seeds inside the array job
SEEDS=(0 42 123 9384 19302)
SEED=\${SEEDS[\$SLURM_ARRAY_TASK_ID]}

EPS_LINF=0.031372549
ALPHA_LINF=0.007843137

log "SLURM_ARRAY_TASK_ID=\$SLURM_ARRAY_TASK_ID -> SEED=\$SEED"
log "Training mode: ${mode}, Architecture: ${arch}."

python model.py \
  --model-file "models/${name}.pth" \
  --force-train \
  --num-epochs 200 \
  --batch-size 256\
  --valid-size 1024 \
  ${ADV_FLAG} \
  --seed "\${SEED}" \
  --arch "${arch}" \
  # --activation "${activation}" \
  --early-stop-metric "balanced" \
  --patience 5 \
  --num-workers 4 \
  --eps-linf "\${EPS_LINF}" \
  --alpha-linf "\${ALPHA_LINF}" \
  --eps-l2 1.0 \
  --alpha-l2 0.25 \
  --train-pgd-steps 10 \
  --test-pgd-steps 15 \
  --eval-every 5 \
  --log-dir "${outdir}"

log "Training for seed=\$SEED done."
EOT