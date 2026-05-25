# PFPTok
PFPTok applies prefix-free parsing (PFP) to construct a dictionary-based tokenizer for genomic sequences. Rather than learning a fixed vocabulary from subword statistics (as BPE and Unigram do), PFPTok partitions sequences into phrases that are prefix-free by construction, producing tokens grounded in the combinatorial structure of the input. This repository contains the experiment code and configurations used to evaluate PFPTok against BPE and Unigram tokenizers across antibiotic resistance classification, genomic benchmark tasks, and hyperparameter ablation studies.

## Repository Structure

```
PFPTok-Experiments/
├── Curated_Genes_Experiment/    # MTB AMR classification (curated genes + whole genome)
│   ├── Data/                    # Phenotypic target labels
│   └── Scripts/
│       ├── Config/              # Training, model, and gene configs
│       ├── src/                 # Models (BERT), tokenizers (PFP/BPE/Unigram), data utils
│       └── main.py              # Entry point
├── Ablation_Study/              # Hyperparameter sweeps for all three tokenizers
│   ├── Sample_Data/             # 10 example MTB isolates (Train/Test splits)
│   ├── Tokenizers/              # PFP, BPE, Unigram tokenizer implementations
│   ├── ablation_utils/          # Data loading and sequence processing
│   ├── main_ablation.py         # Ablation entry point
│   └── run_ablation.sh          # Configurable sweep launcher
├── PFPTok DNALB/                # DNALongBench experiments (eQTL, ETGP)
│   ├── Config/Model_Configs/    # BERT architecture configs
│   ├── src/                     # Models (BERT, HyenaDNA), tokenizers, data utils
│   ├── main.py                  # Entry point
│   ├── submit_slurm.sh          # SLURM job script
│   └── submit_non_slurm.sh      # Local/interactive launch script
├── requirements.txt
└── LICENSE                      # MIT
```

## Installation

```bash
git clone https://github.com/ctestagrose/PFPTok-Experiments.git
cd PFPTok-Experiments
pip install -r requirements.txt
```

> **Note:** The provided `requirements.txt` was generated from an HPC environment and includes CUDA-specific and local editable packages. You may need to install PyTorch separately for your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)) and remove any `-e /path/to/...` or `@ file:///...` lines before installing.

## Usage

PFPTok expects a list of sequences (strings).

**Tokenizer Training:**

```python
from PFPTok.src.PFP_Tokenizer import TokenizerManager

sequences = [
    ["ACGT" * 25, "TGCA" * 25],
    ["AAAA" * 25, "CCCC" * 25],
]

tm = TokenizerManager()
tok = tm.setup_tokenizer(sequences, w=6, d=117)
```

The `w` parameter controls the PFP window size and `d` controls the hash modulus, which together determine phrase boundaries and dictionary granularity.

## Experiments

### 1. Curated MTB Gene Classification

Antibiotic resistance (AMR) classification on *Mycobacterium tuberculosis* isolates using a curated set of resistance-associated genes. Trains a BERT classifier on PFP-, BPE-, or Unigram-tokenized gene sequences and evaluates via cross-validation.

- **Code:** `Curated_Genes_Experiment/Scripts/`
- **Data:** Preprocessed isolate data can be downloaded from the [LLMTB repository](https://github.com/ctestagrose/LLMTB/tree/main/Data). Phenotypic targets are in `Curated_Genes_Experiment/Data/cryptic_targets_all.json`.
- **Configuration:** Edit `Scripts/Config/train_config.json` to set paths, tokenizer type (`pfp`, `bpe`, `unigram`), model hyperparameters, and PFP-specific settings (`pfp_w`, `pfp_d`).

```bash
cd Curated_Genes_Experiment/Scripts
python main.py \
    --sequence_dir /path/to/train_isolates \
    --test_sequence_dir /path/to/test_isolates \
    --gene_file ./Config/Gene_Configs/genes_important.json \
    --target_file ../Data/cryptic_targets_all.json \
    --model_config ./Config/Model_Configs/Base_BERT/base_config_binary.json \
    --tokenizer_type pfp \
    --antibiotic RIF \
    --save_path ./runs/pfp_rif \
    --use_holdout --use_gene_file
```

### 2. Whole Genome Tokenization / Classification

Uses the same pipeline as the curated gene experiments but operates on full genome assemblies (scaffolds) instead of individual genes. Enable whole-genome mode with the `--use_scaffolds` flag.

> **Note:** It is highly recommended to run these experiments with access to a GPU. More vRAM is needed for the full genome experiment.

```bash
python main.py \
    --sequence_dir /path/to/train_isolates \
    --test_sequence_dir /path/to/test_isolates \
    --gene_file ./Config/Gene_Configs/genes_important.json \
    --target_file ../Data/cryptic_targets_all.json \
    --model_config ./Config/Model_Configs/Base_BERT/base_config_binary.json \
    --tokenizer_type pfp \
    --antibiotic RIF \
    --save_path ./runs/pfp_rif_scaffold \
    --use_scaffolds
```

### 3. Ablation Study

Systematic hyperparameter sweeps across all three tokenizers. Measures tokenization statistics (vocab size, compression ratio, token count distributions) and downstream classification performance across parameter grids.

- **Code:** `Ablation_Study/`
- **Sample data:** 10 example MTB isolates are provided in `Sample_Data/` for quick testing.

Update paths in `run_ablation.sh` if needed, then launch a sweep:

```bash
cd Ablation_Study
bash run_ablation.sh <unigram|bpe|pfptok> <quick|focused|comprehensive>
```

Each tokenizer has three preset configurations controlling the size of the parameter grid. These are editable by the user.

Results are saved as JSON, CSV, and a summary text file under `ablation_results/`.

### 4. DNALongBench Experiments

Evaluation on the [DNALongBench](https://github.com/rattlesnakey/DNALongBench) benchmark, covering the eQTL (expression quantitative trait loci) and ETGP (enhancer-target gene prediction) tasks. Supports both PFP-tokenized BERT and HyenaDNA architectures, with ordered and unordered tokenization variants.

> **Note:** It is highly recommended to run these experiments with access to a GPU.


- **Code:** `PFPTok DNALB/`
- **Data:** DNALongBench datasets should be preprocessed into JSON splits (train/validation/test) following the format expected by `--use_json_dataset`.
- **Model configs:** `Config/Model_Configs/Base_BERT/base_config_binary.json`

**Running with SLURM:**

Edit `submit_slurm.sh` to set your account, paths, and partition, then:

```bash
cd "PFPTok DNALB"
sbatch submit_slurm.sh
```

**Running interactively:**

```bash
torchrun --nproc-per-node=1 main.py \
    --use_json_dataset \
    --json_path /path/to/eQTL_splits/Whole_Blood \
    --num_epochs 50 \
    --batch_size 128 \
    --model_type bert \
    --task eQTL \
    --model_config ./Config/Model_Configs/Base_BERT/base_config_binary.json \
    --antibiotic binary_json \
    --save_path ./runs/eqtl_whole_blood
```

Add `--ordered` to use ordered tokenization. To run with HyenaDNA instead:

```bash
torchrun --nproc-per-node=1 main.py \
    --use_json_dataset \
    --json_path /path/to/eQTL_splits/Whole_Blood \
    --model_type hyena \
    --task eQTL \
    --model_config ./Config/Model_Configs/Base_BERT/base_config_binary.json \
    --antibiotic binary_json \
    --save_path ./runs/eqtl_hyena
```

The eQTL task runs across 9 tissue splits: Adipose Subcutaneous, Artery Tibial, Cultured Fibroblasts, Muscle Skeletal, Nerve Tibial, Skin (Not Sun Exposed), Skin (Sun Exposed), Thyroid, and Whole Blood.
