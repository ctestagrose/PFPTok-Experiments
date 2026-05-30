#!/bin/sh
#SBATCH --job-name=pfptok_dnalongbench
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/%j_disp.log
#SBATCH --error=slurm_logs/%j_disp.err

# CONFIGURATION: edit these paths before submitting
# Root directory containing per-tissue eQTL split folders,
# or the single ETGP folder. Each split folder must contain
# <NAME>_train.jsonl.gz, <NAME>_valid.jsonl.gz, <NAME>_test.jsonl.gz
JSON_ROOT="/path/to/your/DNALongBench/data"

# Model config (no edit needed if running from dnalongbench/)
MODEL_CONFIG="./config/model_configs/base_bert/base_config_binary.json"

# Where to write results. Two subdirs will be created:
# one for unordered tokenization, one for ordered.
SAVE_ROOT_UNORDERED="./runs/eqtl_unordered"
SAVE_ROOT_ORDERED="./runs/eqtl_ordered"

# Task: eQTL or ETGP
TASK="eQTL"

# Model type: bert or hyena
MODEL_TYPE="bert"

# Activate your environment here (adjust for conda/venv/module):
# module load cuda
# conda activate your_env

set -euo pipefail
export TQDM_DISABLE=1

if [[ "$TASK" == "ETGP" ]]; then
  SPLITS=(
    "ETGP"
  )
else
  SPLITS=(
    "Adipose_Subcutaneous"
    "Artery_Tibial"
    "Cells_Cultured_fibroblasts"
    "Muscle_Skeletal"
    "Nerve_Tibial"
    "Skin_Not_Sun_Exposed_Suprapubic"
    "Skin_Sun_Exposed_Lower_leg"
    "Thyroid"
    "Whole_Blood"
  )
fi

for name in "${SPLITS[@]}"; do
  if [[ "$TASK" == "ETGP" ]]; then
    json_path="${JSON_ROOT}"
  else
    json_path="${JSON_ROOT}/${name}"
  fi

  echo "--- ${name} ---"

  mkdir -p "${SAVE_ROOT_UNORDERED}/${name}/"
  torchrun --nproc-per-node=1 --master-port=12875 main.py \
    --use_json_dataset \
    --json_path "${json_path}/${name}" \
    --num_epochs 20 \
    --batch_size 128 \
    --model_type "${MODEL_TYPE}" \
    --task "${TASK}" \
    --model_config "${MODEL_CONFIG}" \
    --antibiotic "binary_json" \
    --save_path "${SAVE_ROOT_UNORDERED}/${name}"

  mkdir -p "${SAVE_ROOT_ORDERED}/${name}/"
  torchrun --nproc-per-node=1 --master-port=12875 main.py \
    --use_json_dataset \
    --json_path "${json_path}/${name}" \
    --num_epochs 20 \
    --batch_size 128 \
    --model_type "${MODEL_TYPE}" \
    --task "${TASK}" \
    --model_config "${MODEL_CONFIG}" \
    --antibiotic "binary_json" \
    --save_path "${SAVE_ROOT_ORDERED}/${name}" \
    --ordered
done
