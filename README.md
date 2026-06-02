# PFPTok

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![CI](https://github.com/ctestagrose/PFPTok-Experiments/actions/workflows/ci.yml/badge.svg)](https://github.com/ctestagrose/PFPTok-Experiments/actions/workflows/ci.yml)

PFPTok applies prefix-free parsing (PFP) to construct a dictionary-based tokenizer for genomic sequences. Rather than learning a fixed vocabulary from subword statistics (as BPE and Unigram do), PFPTok partitions sequences into phrases that are prefix-free by construction, producing tokens grounded in the combinatorial structure of the input. This repository contains the experiment code and configurations used to evaluate PFPTok against BPE and Unigram tokenizers across antibiotic resistance classification, genomic benchmark tasks, and hyperparameter ablation studies.

## Repository Structure

```
PFPTok-Experiments/
├── curated_genes/               # MTB AMR classification (curated genes + whole genome)
│   ├── data/                    # Phenotypic target labels
│   ├── config/
│   │   ├── gene_configs/        # Per-antibiotic resistance gene lists
│   │   ├── model_configs/       # BERT architecture configs
│   │   └── train_config.json    # Default training config
│   ├── models/                  # BERT classifier implementation
│   ├── tokenizers/              # PFP, BPE, Unigram tokenizer implementations
│   ├── utils/                   # Data loading, dataset, metrics, loss
│   └── main.py                  # Entry point
├── ablation/                    # Hyperparameter sweeps for all three tokenizers
│   ├── tokenizers/              # PFP, BPE, Unigram tokenizer implementations
│   ├── utils/                   # Data loading and sequence processing
│   ├── main.py                  # Ablation entry point
│   └── run_ablation.sh          # Configurable sweep launcher
├── dnalongbench/                # DNALongBench experiments (eQTL, ETGP)
│   ├── config/model_configs/    # BERT architecture configs
│   ├── models/                  # BERT and HyenaDNA implementations
│   ├── utils/                   # Data utils, tokenizers, metrics
│   ├── main.py                  # Entry point
│   ├── submit_slurm.sh          # SLURM job script
│   └── submit_non_slurm.sh      # Local/interactive launch script
├── sample_data/                 # Example MTB isolates for quick testing
│   ├── curated_genes_sample_data/
│   └── abalation_and_whole_genome_sample_data/
├── tests/                       # pytest test suite
├── requirements.txt
├── requirements-test.txt        # Minimal deps for running tests (no GPU required)
├── environment.yml
└── LICENSE                      # MIT
```

## Installation

**Recommended - conda environment (includes PyTorch + CUDA):**

```bash
git clone https://github.com/ctestagrose/PFPTok-Experiments.git
cd PFPTok-Experiments
conda env create -f environment.yml
conda activate pfp-tok
```

**Alternative - pip only:**

```bash
git clone https://github.com/ctestagrose/PFPTok-Experiments.git
cd PFPTok-Experiments
python3 -m venv .venv
source .venv/bin/activate
pip install - requirements.txt
```

> **Note:** `requirements.txt` was generated from an HPC environment and includes CUDA-specific packages (`+cu128` version suffixes). You may need to install PyTorch separately for your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)) and strip those suffixes before installing on CPU-only machines.

## Running the Tests

The test suite requires no GPU and uses the bundled `sample_data/`:

```bash
pip install -r requirements-test.txt pytest
pytest tests/ -v
```

## Configuring Paths

Each experiment reads its paths from a dedicated config file or shell script. The defaults point to the bundled `sample_data/` so you can run a quick smoke test immediately. Replace them with your full dataset paths before running real experiments.

| Experiment | File to edit | Key fields |
|---|---|---|
| Curated genes / Whole genome | `curated_genes/config/train_config.json` | `sequence_dir`, `test_sequence_dir`, `save_path` |
| Ablation study | `ablation/run_ablation.sh` (top of file) | `SEQUENCE_DIR`, `TEST_SEQUENCE_DIR`, `TARGET_FILE` |
| DNALongBench (interactive) | `dnalongbench/submit_non_slurm.sh` (top of file) | `JSON_ROOT`, `SAVE_ROOT_UNORDERED`, `SAVE_ROOT_ORDERED`, `TASK`, `MODEL_TYPE` |
| DNALongBench (SLURM) | `dnalongbench/submit_slurm.sh` (top of file) | Same as above, plus SBATCH directives |

### Curated genes — `train_config.json`

```json
{
    "sequence_dir": "/path/to/your/train_isolates",
    "test_sequence_dir": "/path/to/your/test_isolates",
    "save_path": "./runs/my_experiment",
    "antibiotic": "RIF",
    "tokenizer_type": "pfp",
    ...
}
```

Set `antibiotic` to one of: `RIF`, `INH`, `EMB`, `RFB`, `LEV`, `MXF`, `KAN`, `AMI`, `DLM`, `LZD`, `CFZ`, `BDQ`, or `All` for multi-drug. Set `tokenizer_type` to `pfp`, `bpe`, or `unigram`. All other paths (gene config, model config, target labels) are relative to `curated_genes/` and require no changes.

For whole-genome mode, also set `"use_scaffolds": true` and `"use_gene_file": false`.

### Ablation — `run_ablation.sh`

Edit the three lines at the top of the file:

```bash
SEQUENCE_DIR="/path/to/your/train_isolates"
TEST_SEQUENCE_DIR="/path/to/your/test_isolates"
TARGET_FILE="/path/to/cryptic_targets_all.json"
```

### DNALongBench — `submit_non_slurm.sh` / `submit_slurm.sh`

Edit the variables at the top of the script:

```bash
JSON_ROOT="/path/to/your/DNALongBench/data"   # folder containing per-tissue split dirs
SAVE_ROOT_UNORDERED="./runs/eqtl_unordered"
SAVE_ROOT_ORDERED="./runs/eqtl_ordered"
TASK="eQTL"          # or ETGP
MODEL_TYPE="bert"    # or hyena
```

For SLURM, also fill in your cluster account, partition, and email in the `#SBATCH` header.

## Usage

PFPTok expects a list of sequences (strings).

**Tokenizer Training:**

```python
from curated_genes.tokenizers.pfp_tokenizer import TokenizerManager

sequences = [
    ["ACGT" * 25, "TGCA" * 25],
    ["AAAA" * 25, "CCCC" * 25],
]

tm = TokenizerManager()
tok = tm.setup_tokenizer(sequences, w=6, p=117)
```

The `w` parameter controls the PFP window size and `p` controls the hash period, which together determine phrase boundaries and dictionary granularity.

## Experiments

### 1. Curated MTB Gene Classification

Antibiotic resistance (AMR) classification on *Mycobacterium tuberculosis* isolates using a curated set of resistance-associated genes. Trains a BERT classifier on PFP-, BPE-, or Unigram-tokenized gene sequences and evaluates via cross-validation.

- **Code:** `curated_genes/`
- **Data:** Preprocessed isolate data can be downloaded from the [LLMTB repository](https://github.com/ctestagrose/LLMTB/tree/main/Data). Phenotypic targets are in `curated_genes/data/cryptic_targets_all.json`.
- **Configuration:** All settings are read from `curated_genes/config/train_config.json` — no command-line flags are needed. Edit that file to set data paths, tokenizer type (`pfp`, `bpe`, `unigram`), model hyperparameters, and PFP-specific parameters (`pfp_w`, `pfp_p`). Key fields for this experiment:

```json
{
    "use_scaffolds": false,
    "use_gene_file": true,
    "tokenizer_type": "pfp",
    "pfp_w": 6,
    "pfp_p": 117
}
```

```bash
cd curated_genes
python main.py
```

### 2. Whole Genome Tokenization / Classification

Uses the same pipeline and entry point as the curated gene experiment but operates on full genome assemblies instead of individual genes. All configuration is done through `curated_genes/config/train_config.json`.

> **Note:** It is highly recommended to run these experiments with access to a GPU. More vRAM is needed for the full genome experiment.

Set the following fields in `curated_genes/config/train_config.json` to switch to whole-genome mode:

```json
{
    "use_scaffolds": true,
    "use_gene_file": false,
    "tokenizer_type": "pfp",
    "pfp_w": 100,
    "pfp_p": 4096
}
```

Update `sequence_dir` and `test_sequence_dir` to point to your genome assemblies (see [Configuring Paths](#configuring-paths)), then run:

```bash
cd curated_genes
python main.py
```

### 3. Ablation Study

Systematic hyperparameter sweeps across all three tokenizers. Measures tokenization statistics (vocab size, compression ratio, token count distributions) and downstream classification performance across parameter grids.

- **Code:** `ablation/`
- **Sample data:** 10 example MTB isolates are provided in `sample_data/` for quick testing.

Update paths in `ablation/run_ablation.sh` if needed, then launch a sweep:

```bash
cd ablation
bash run_ablation.sh <unigram|bpe|pfptok> <quick|focused|comprehensive>
```

Each tokenizer has three preset configurations controlling the size of the parameter grid. These are editable by the user.

Results are saved as JSON, CSV, and a summary text file under `ablation_results/`.

### 4. DNALongBench Experiments

Evaluation on the [DNALongBench](https://github.com/wenduocheng/DNALongBench) benchmark, covering the eQTL (expression quantitative trait loci) and ETGP (enhancer-target gene prediction) tasks. Supports both PFP-tokenized BERT and HyenaDNA architectures, with ordered and unordered tokenization variants.

> **Note:** It is highly recommended to run these experiments with access to a GPU.
> 


- **Code:** `dnalongbench/`
- **Data:** DNALongBench datasets should be preprocessed into JSON splits (train/validation/test) following the format expected by `--use_json_dataset`.
- **Model configs:** `dnalongbench/config/model_configs/base_bert/base_config_binary.json`

**Running with SLURM:**

Edit `dnalongbench/submit_slurm.sh` to set your account, paths, and partition, then:

```bash
cd dnalongbench
sbatch submit_slurm.sh
```

**Running interactively:**

```bash
cd dnalongbench
bash submit_non_slurm.sh
```

To run with HyenaDNA instead, set `MODEL_TYPE="hyena"` at the top of the script before running.

The eQTL task runs across 9 tissue splits: Adipose Subcutaneous, Artery Tibial, Cultured Fibroblasts, Muscle Skeletal, Nerve Tibial, Skin (Not Sun Exposed), Skin (Sun Exposed), Thyroid, and Whole Blood.
