#!/bin/sh
#SBATCH --job-name=dnalongbenchberteqtlnoval
#SBATCH --account=simone.marini
#SBATCH --qos=simone.marini
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=testagroseconrad@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gres=gpu:b200:1
#SBATCH --partition=hpg-b200
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/%j_disp.log
#SBATCH --error=slurm_logs/%j_disp.err


module load cuda
module load mamba 

mamba activate hyena-dna

set -euo pipefail

export TQDM_DISABLE=1

JSON_ROOT="/blue/boucher/testagroseconrad/DNALONGBENCH/Data/eQTL_Splits_DNALBFF"
MODEL_CONFIG="/blue/boucher/testagroseconrad/DNALONGBENCH/Experiments/Config/Model_Configs/Base_BERT/base_config_binary.json"

SAVE_ROOT_UNORDERED="/blue/boucher/testagroseconrad/DNALONGBENCH/Experiments/runs/Faithful_Runs/eQTL_PFPBERT_w20_d4090_Unordered_Faithful_Hyena_Comp_All_No_Validation"
SAVE_ROOT_ORDERED="/blue/boucher/testagroseconrad/DNALONGBENCH/Experiments/runs/Faithful_Runs/eQTL_PFPBERT_w20_d4090_Ordered_Faithful"

TASK="eQTL" 

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
    --model_type "bert" \
    --task "${TASK}" \
    --model_config "${MODEL_CONFIG}" \
    --antibiotic "binary_json" \
    --save_path "${SAVE_ROOT_UNORDERED}/${name}" \

  mkdir -p "${SAVE_ROOT_ORDERED}/${name}/"
  torchrun --nproc-per-node=1 --master-port=12875 main.py \
    --use_json_dataset \
    --json_path "${json_path}/${name}" \
    --num_epochs 20 \
    --batch_size 128 \
    --model_type "bert" \
    --task "${TASK}" \
    --model_config "${MODEL_CONFIG}" \
    --antibiotic "binary_json" \
    --save_path "${SAVE_ROOT_ORDERED}/${name}" \
    --ordered 
done
