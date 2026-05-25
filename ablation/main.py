import argparse
import json
import os
import glob
import time
import psutil
import random
import threading
from datetime import datetime
import pandas as pd
import numpy as np
import traceback
import gc

from utils.data_prep import DataPreparer
from utils.sequence_processor import SequenceProcessor


def _import_unigram():
    from tokenizers.unigram_tokenizer import TokenizerManagerUnigram
    return TokenizerManagerUnigram


def _import_bpe():
    from tokenizers.bpe_tokenizer import TokenizerManagerBPE
    return TokenizerManagerBPE


def _import_pfptok():
    from tokenizers.pfp_tokenizer import TokenizerManager
    return TokenizerManager


TOKENIZER_IMPORTS = {
    "unigram": _import_unigram,
    "bpe": _import_bpe,
    "pfptok": _import_pfptok,
}


# Argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Unified Tokenizer Ablation Study",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--tokenizer_type", type=str, required=True,
                        choices=["unigram", "bpe", "pfptok"],
                        help="Which tokenizer family to ablate")
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--test_sequence_dir", type=str, required=True)
    parser.add_argument("--target_file", type=str, required=True)
    parser.add_argument("--antibiotic", type=str, default="RIF")
    parser.add_argument("--output_dir", type=str, default="./ablation_results")
    parser.add_argument("--num_sequences", type=int, default=None,
                        help="Max sequences to load (None = use all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_scaffolds", action="store_true")
    parser.add_argument("--scaffold_concat", action="store_true")
    parser.add_argument("--isolate_id_from_dir", action="store_true")
    parser.add_argument("--Kmer_Size", type=int, default=31)
    parser.add_argument("--stride", type=int, default=1)

    # Unigram-specific
    parser.add_argument("--vocab_sizes", type=int, nargs="+", default=[10000, 50000, 100000])
    parser.add_argument("--max_sentencepiece_lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--model_types", type=str, nargs="+", default=["unigram"],
                        help="SentencePiece model types")
    parser.add_argument("--chunk_sizes", type=int, nargs="+", default=[10000, 20000, 50000])

    # BPE-specific
    parser.add_argument("--min_frequencies", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--max_total_chars_list", type=int, nargs="+",
                        default=[10_000_000, 20_000_000, 50_000_000])
    parser.add_argument("--window_sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--stride_sizes", type=int, nargs="+", default=[1000, 10000, 100000])

    # PFPTok-specific
    parser.add_argument("--w_values", type=int, nargs="+", default=[3, 5, 10, 15, 20])
    parser.add_argument("--d_values", type=int, nargs="+", default=[31, 63, 127, 255, 511])

    return parser.parse_args()


# File Discovery
def get_isolate_scaffold_files(sequence_dir, num_sequences=None):
    files = []
    for isolate in sorted(os.listdir(sequence_dir)):
        iso_dir = os.path.join(sequence_dir, isolate)
        if not os.path.isdir(iso_dir):
            continue
        f = os.path.join(iso_dir, f"{isolate}.H37Rv_ordered.fasta")
        if os.path.exists(f):
            files.append(f)
        if num_sequences and len(files) >= num_sequences:
            break
    print(f"Found {len(files)} sequence files")
    return files


def _get_files(args, sequence_dir):
    if args.use_scaffolds:
        return get_isolate_scaffold_files(sequence_dir, args.num_sequences)
    files = glob.glob(os.path.join(sequence_dir, "**", "*.fasta"), recursive=True)
    if args.num_sequences:
        random.seed(args.seed)
        files = random.sample(files, min(args.num_sequences, len(files)))
    return files


# Memory profiling
def get_peak_memory_increase(func, *args, **kwargs):
    gc.collect()
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss / 1024 / 1024

    class Monitor:
        def __init__(self):
            self.peak = baseline
            self.running = True

        def run(self):
            while self.running:
                self.peak = max(self.peak, process.memory_info().rss / 1024 / 1024)
                time.sleep(0.1)

    mon = Monitor()
    t = threading.Thread(target=mon.run, daemon=True)
    t.start()
    try:
        result = func(*args, **kwargs)
        time.sleep(0.2)
        mon.running = False
        t.join(timeout=1)
        return result, max(0, mon.peak - baseline)
    except Exception:
        mon.running = False
        raise


# Data loading  (this is a shared pipeline for all tokenizer types)
def load_data(args):
    print("\n" + "=" * 60)
    print(f"Loading Data for {args.tokenizer_type.upper()} ablation.")
    print("=" * 60)

    train_files = _get_files(args, args.sequence_dir)
    test_files = _get_files(args, args.test_sequence_dir)
    print(f"Train files: {len(train_files)}")
    print(f"Test files:  {len(test_files)}")

    sequence_processor = SequenceProcessor(args.Kmer_Size, args.stride)

    TokenizerCls = TOKENIZER_IMPORTS[args.tokenizer_type]()
    tokenizer_manager = TokenizerCls()

    class _TempArgs:
        pass

    temp_args = _TempArgs()
    for attr in ("use_scaffolds", "scaffold_concat", "isolate_id_from_dir",
                 "Kmer_Size", "stride"):
        setattr(temp_args, attr, getattr(args, attr, False))

    dp = DataPreparer(sequence_processor, tokenizer_manager, temp_args)

    print("\nPreparing training data...")
    train_full_seqs = dp.prep_data(
        fasta_files=train_files,
        target_file=args.target_file,
        target_format=args.antibiotic,
        mode="Train",
    )

    print("\nPreparing test data...")
    test_full_seqs = dp.prep_data(
        fasta_files=test_files,
        target_file=args.target_file,
        target_format=args.antibiotic,
        mode="Evaluate",
    )

    train_genomes = [seq[0] for seq in train_full_seqs if seq and seq[0]]
    test_genomes = [seq[0] for seq in test_full_seqs if seq and seq[0]]

    print(f"\nLoaded {len(train_genomes)} training genomes")
    print(f"Loaded {len(test_genomes)} test genomes")
    print(f" * Total train bp: {sum(len(g) for g in train_genomes):,}")
    print(f" * Total test  bp: {sum(len(g) for g in test_genomes):,}")

    return {
        "train_full_seqs": train_full_seqs,  # for BPE/PFPTok
        "test_full_seqs": test_full_seqs,
        "train_genomes": train_genomes,  # for Unigram setup & all evals
        "test_genomes": test_genomes,
        "sequence_processor": sequence_processor,
    }


# Config generation  (one function per tokenizer type)
def _gen_configs_unigram(args):
    cfgs = []
    for vs in args.vocab_sizes:
        for ml in args.max_sentencepiece_lengths:
            for mt in args.model_types:
                for cs in args.chunk_sizes:
                    cfgs.append(dict(vocab_size=vs, max_sentencepiece_length=ml,
                                     model_type=mt, chunk_size=cs))
    return cfgs


def _gen_configs_bpe(args):
    cfgs = []
    for vs in args.vocab_sizes:
        for mf in args.min_frequencies:
            for mc in args.max_total_chars_list:
                for w in args.window_sizes:
                    for s in args.stride_sizes:
                        if s <= w:
                            cfgs.append(dict(vocab_size=vs, min_frequency=mf,
                                             max_total_chars=mc, window=w, stride=s))
    return cfgs


def _gen_configs_pfptok(args):
    return [dict(w=w, d=d) for w in args.w_values for d in args.d_values]


GENERATE_CONFIGS = {
    "unigram": _gen_configs_unigram,
    "bpe": _gen_configs_bpe,
    "pfptok": _gen_configs_pfptok,
}


# Per-config runners
# UNIGRAM
def _run_unigram(config, data, args):
    TokenizerCls = TOKENIZER_IMPORTS["unigram"]()
    train_genomes = data["train_genomes"]
    test_genomes = data["test_genomes"]

    def train_fn():
        tm = TokenizerCls(vocab_size=config["vocab_size"],
                          chunk_size=config["chunk_size"])
        tokenizer = tm.setup_tokenizer(
            sequences=train_genomes,
            vocab_size=config["vocab_size"],
            max_sentencepiece_length=config["max_sentencepiece_length"],
            model_type=config["model_type"],
            max_sentence_length=5000000,
            max_training_sequences=10000,
            use_chunking=False,
        )
        return tm, tokenizer

    t0 = time.time()
    (tm, tokenizer), mem = get_peak_memory_increase(train_fn)
    training_time = time.time() - t0

    results = {
        "training_time_sec": training_time,
        "memory_used_mb": mem,
        "actual_vocab_size": tokenizer.vocab_size,
    }

    train_m = _eval_unigram(train_genomes, tm, tokenizer, "train")
    test_m = _eval_unigram(test_genomes, tm, tokenizer, "test")
    _merge_metrics(results, train_m, test_m)
    return results


def _eval_unigram(genomes, tm, tokenizer, label):
    token_counts, unk_counts = [], []
    total_tokens = total_bp = 0
    for idx, gs in enumerate(genomes):
        try:
            ids, _, _ = tm.encode_sequence_chunked(
                sequence=gs, tokenizer=tokenizer,
                split="test", dropout_prob=0.0, use_chunking=True,
            )
            n = len(ids)
            u = sum(1 for t in ids if t == tokenizer.unk_token_id)
            token_counts.append(n)
            unk_counts.append(u)
            total_tokens += n
            total_bp += len(gs)
            if idx < 3:
                print(f"[{label}] Genome {idx}: {len(gs):,} bp -> {n:,} tokens ({u} UNK)")
        except Exception as e:
            print(f"Warning: encode failed for {label} seq {idx}: {e}")
    if not token_counts:
        return {}
    return {
        "avg_tokens": float(np.mean(token_counts)),
        "total_tokens": total_tokens,
        "total_bp": total_bp,
        "compression_ratio": total_bp / total_tokens if total_tokens else 0,
        "unk_percentage": (sum(unk_counts) / total_tokens * 100) if total_tokens else 0,
    }


# BPE
def _run_bpe(config, data, args):
    TokenizerCls = TOKENIZER_IMPORTS["bpe"]()
    train_full = data["train_full_seqs"]
    train_genomes = data["train_genomes"]
    test_genomes = data["test_genomes"]

    def train_fn():
        tm = TokenizerCls(vocab_size=config["vocab_size"])
        tok = tm.setup_tokenizer(
            sequences=train_full,
            vocab_size=config["vocab_size"],
            min_frequency=config["min_frequency"],
            max_total_chars=config["max_total_chars"],
            window=config["window"],
            stride=config["stride"],
        )
        return tm, tok

    t0 = time.time()
    (tm, tokenizer), mem = get_peak_memory_increase(train_fn)
    training_time = time.time() - t0

    results = {
        "training_time_sec": training_time,
        "memory_used_mb": mem,
        "actual_vocab_size": tokenizer.vocab_size,
    }

    train_m = _eval_bpe(train_genomes, tokenizer, "train")
    test_m = _eval_bpe(test_genomes, tokenizer, "test")
    _merge_metrics(results, train_m, test_m)
    return results


def _eval_bpe(genomes, tokenizer, label):
    total_tokens = total_chars = unk_count = 0
    unk_token = tokenizer.unk_token

    for idx, genome in enumerate(genomes):
        tokens = tokenizer.tokenize(genome)
        n = len(tokens)
        u = sum(1 for t in tokens if t == unk_token)
        total_tokens += n
        total_chars += len(genome)
        unk_count += u

    return {
        "avg_tokens": total_tokens / len(genomes) if genomes else 0,
        "compression_ratio": total_chars / total_tokens if total_tokens else 0,
        "unk_percentage": (unk_count / total_tokens * 100) if total_tokens else 0,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
    }


# PFPTOK
def _run_pfptok(config, data, args):
    TokenizerCls = TOKENIZER_IMPORTS["pfptok"]()
    train_full = data["train_full_seqs"]
    test_full = data["test_full_seqs"]

    def train_fn():
        tm = TokenizerCls()
        tok = tm.setup_tokenizer(sequences=train_full, w=config["w"], d=config["d"])
        return tm, tok

    t0 = time.time()
    (tm, tokenizer), mem = get_peak_memory_increase(train_fn)
    training_time = time.time() - t0

    actual_vocab = len(tokenizer.get_vocab())
    results = {
        "training_time_sec": training_time,
        "memory_used_mb": mem,
        "actual_vocab_size": actual_vocab,
    }

    train_m = _eval_pfptok(train_full, tm, "train")
    test_m = _eval_pfptok(test_full, tm, "test")
    _merge_metrics(results, train_m, test_m)

    results["train_tokens_per_bp"] = (1.0 / train_m["compression_ratio"]
                                      if train_m.get("compression_ratio") else 0)
    results["test_tokens_per_bp"] = (1.0 / test_m["compression_ratio"]
                                     if test_m.get("compression_ratio") else 0)
    return results


def _eval_pfptok(full_seqs, tm, label):
    encoded_ids, (unk_count, non_unk_count) = tm.encode_sequences(sequences=full_seqs)

    total_chars = sum(len(g) for seq in full_seqs for g in seq if isinstance(g, str))
    total_tokens = len(encoded_ids)
    total_token_count = unk_count + non_unk_count

    print(f"[{label}] {len(full_seqs)} genomes -> {total_tokens:,} tokens "
          f"({unk_count} UNK / {total_token_count} total)")

    return {
        "avg_tokens": total_tokens / len(full_seqs) if full_seqs else 0,
        "compression_ratio": total_chars / total_tokens if total_tokens else 0,
        "unk_percentage": (unk_count / total_token_count * 100) if total_token_count else 0,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
    }


# Shared metric merge + generalization diff
def _merge_metrics(results, train_m, test_m):
    for k, v in train_m.items():
        results[f"train_{k}"] = v
    for k, v in test_m.items():
        results[f"test_{k}"] = v
    results["compression_ratio_diff"] = abs(
        train_m.get("compression_ratio", 0) - test_m.get("compression_ratio", 0))
    results["unk_percentage_diff"] = abs(
        train_m.get("unk_percentage", 0) - test_m.get("unk_percentage", 0))


RUN_SINGLE = {
    "unigram": _run_unigram,
    "bpe": _run_bpe,
    "pfptok": _run_pfptok,
}


# Shared runner wrapper
def run_single_config(tokenizer_type, config, data, args):
    print("\n" + "=" * 60)
    print("Testing Configuration:")
    for k, v in config.items():
        print(f" * {k}: {v}")
    print("=" * 60)

    result = {
        "config": config.copy(),
        "tokenizer_type": tokenizer_type,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        metrics = RUN_SINGLE[tokenizer_type](config, data, args)
        result.update(metrics)
        result["status"] = "success"

        print("\n" + "-" * 60)
        print("Results Summary:")
        print(f" * Training time: {result.get('training_time_sec', 0):.2f}s")
        print(f" * Memory used:   {result.get('memory_used_mb', 0):.2f} MB")
        if "actual_vocab_size" in result:
            print(f" * Vocab size:    {result['actual_vocab_size']:,}")
        print(f" * Train compression: {result.get('train_compression_ratio', 0):.2f} bp/token")
        print(f" * Test  compression: {result.get('test_compression_ratio', 0):.2f} bp/token")
        print(f" * Train UNK%: {result.get('train_unk_percentage', 0):.2f}%")
        print(f" * Test  UNK%: {result.get('test_unk_percentage', 0):.2f}%")
        print("-" * 60)

    except Exception as e:
        print(f"\n{'!' * 60}")
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        print(f"{'!' * 60}")
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    finally:
        gc.collect()

    return result


# Serialization helpers
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


# Result persistence
_PRIORITY_COLS = {
    "unigram": [
        "vocab_size", "max_sentencepiece_length", "model_type", "chunk_size",
        "training_time_sec", "memory_used_mb",
        "train_avg_tokens", "train_compression_ratio", "train_unk_percentage",
        "test_avg_tokens", "test_compression_ratio", "test_unk_percentage",
        "compression_ratio_diff", "unk_percentage_diff", "status",
    ],
    "bpe": [
        "vocab_size", "min_frequency", "max_total_chars", "window", "stride",
        "training_time_sec", "memory_used_mb",
        "train_avg_tokens", "train_compression_ratio", "train_unk_percentage",
        "test_avg_tokens", "test_compression_ratio", "test_unk_percentage",
        "compression_ratio_diff", "unk_percentage_diff", "status",
    ],
    "pfptok": [
        "w", "d", "actual_vocab_size",
        "training_time_sec", "memory_used_mb",
        "train_avg_tokens", "train_compression_ratio", "train_unk_percentage",
        "test_avg_tokens", "test_compression_ratio", "test_unk_percentage",
        "compression_ratio_diff", "unk_percentage_diff", "status",
    ],
}


def save_pfptok_nt_per_token(results, output_dir):
    rows = []
    for r in results:
        if r.get("status") != "success":
            continue
        cfg = r["config"]
        rows.append({
            "w": cfg["w"],
            "d": cfg["d"],
            "vocab_size": r.get("actual_vocab_size"),
            "train_nt_per_token": round(r.get("train_compression_ratio", 0), 4),
            "test_nt_per_token": round(r.get("test_compression_ratio", 0), 4),
            "train_test_diff": round(r.get("compression_ratio_diff", 0), 4),
        })
    rows.sort(key=lambda x: (x["w"], x["d"]))
    path = os.path.join(output_dir, "pfptok_nt_per_token.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f" * Saved nt/token summary -> {path}")


def save_results(results, output_dir, tokenizer_type):
    os.makedirs(output_dir, exist_ok=True)
    ser = convert_to_serializable(results)
    tag = tokenizer_type

    jp = os.path.join(output_dir, f"ablation_results_{tag}.json")
    with open(jp, "w") as f:
        json.dump(ser, f, indent=2)
    print(f"\n * Saved JSON  -> {jp}")

    flat = []
    for r in ser:
        d = r.copy()
        if "config" in d:
            d.update(d.pop("config"))
        flat.append(d)
    df = pd.DataFrame(flat)
    prio = _PRIORITY_COLS.get(tag, [])
    cols = [c for c in prio if c in df.columns] + [c for c in df.columns if c not in prio]
    df = df[cols]
    cp = os.path.join(output_dir, f"ablation_results_{tag}.csv")
    df.to_csv(cp, index=False)
    print(f" * Saved CSV   -> {cp}")

    sp = os.path.join(output_dir, f"summary_{tag}.txt")
    successful = [r for r in results if r.get("status") == "success"]
    with open(sp, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"{tag.upper()} Ablation Study Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total configs tested: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(results) - len(successful)}\n\n")
        if successful:
            best = max(successful, key=lambda x: x.get("test_compression_ratio", 0))
            fastest = min(successful, key=lambda x: x.get("training_time_sec", float("inf")))
            f.write(f"Best test compression:\n  Config: {best['config']}\n")
            f.write(f" * Compression: {best['test_compression_ratio']:.2f} bp/token\n\n")
            f.write(f"Fastest training:\n  Config: {fastest['config']}\n")
            f.write(f" * Time: {fastest['training_time_sec']:.2f}s\n\n")

            if tag == "pfptok":
                smallest = min(successful, key=lambda x: x.get("actual_vocab_size", float("inf")))
                largest = max(successful, key=lambda x: x.get("actual_vocab_size", 0))
                f.write(f"Smallest vocab:\n  Config: {smallest['config']}\n")
                f.write(f" * Vocab size: {smallest['actual_vocab_size']:,}\n\n")
                f.write(f"Largest vocab:\n  Config: {largest['config']}\n")
                f.write(f" * Vocab size: {largest['actual_vocab_size']:,}\n\n")

            if "compression_ratio_diff" in best:
                best_gen = min(successful, key=lambda x: x.get("compression_ratio_diff", float("inf")))
                f.write(f"Best generalization (smallest compression diff):\n")
                f.write(f" * Config: {best_gen['config']}\n")
                f.write(f" * Diff: {best_gen['compression_ratio_diff']:.4f}\n")
    print(f" * Saved summary -> {sp}")


# Comparison table
def print_comparison_table(results, tokenizer_type):
    ok = [r for r in results if r.get("status") == "success"]
    if not ok:
        print("\nNo successful results to display")
        return

    print("\n" + "=" * 80)
    print(f"{tokenizer_type.upper()} Results Comparison Table")
    print("=" * 80)

    rows = []
    for r in ok:
        row = {}
        cfg = r["config"]
        for k, v in cfg.items():
            row[k] = f"{v:,}" if isinstance(v, int) and v > 999 else v
        row["Time(s)"] = f"{r.get('training_time_sec', 0):.1f}"
        row["Mem(MB)"] = f"{r.get('memory_used_mb', 0):.0f}"
        if "actual_vocab_size" in r:
            row["Vocab"] = f"{r['actual_vocab_size']:,}"
        row["Train_BP/Tok"] = f"{r.get('train_compression_ratio', 0):.2f}"
        row["Test_BP/Tok"] = f"{r.get('test_compression_ratio', 0):.2f}"
        row["Train_UNK%"] = f"{r.get('train_unk_percentage', 0):.2f}"
        row["Test_UNK%"] = f"{r.get('test_unk_percentage', 0):.2f}"
        if "compression_ratio_diff" in r:
            row["Diff"] = f"{r['compression_ratio_diff']:.2f}"
        rows.append(row)

    print(pd.DataFrame(rows).to_string(index=False))
    print("=" * 80)


# Main
def main():
    args = parse_arguments()
    tt = args.tokenizer_type

    print("\n" + "=" * 80)
    print(f"Tokenizer Ablation Study  --  {tt.upper()}")
    print("=" * 80)

    print("\nStudy Configuration:")
    if tt == "unigram":
        print(f" * Vocab sizes:    {args.vocab_sizes}")
        print(f" * Max SP lengths: {args.max_sentencepiece_lengths}")
        print(f" * Model types:    {args.model_types}")
        print(f" * Chunk sizes:    {args.chunk_sizes}")
    elif tt == "bpe":
        print(f" * Vocab sizes:      {args.vocab_sizes}")
        print(f" * Min frequencies:  {args.min_frequencies}")
        print(f" * Max total chars:  {args.max_total_chars_list}")
        print(f" * Window sizes:     {args.window_sizes}")
        print(f" * Stride sizes:     {args.stride_sizes}")
    elif tt == "pfptok":
        print(f" * Window sizes (w): {args.w_values}")
        print(f" * Period values (d): {args.d_values}")
    print(f" * Num sequences:  {args.num_sequences or 'all'}")
    print(f" * Output directory: {args.output_dir}")

    configs = GENERATE_CONFIGS[tt](args)
    print(f"\nTotal configurations to test: {len(configs)}")

    print("\nLoading data...")
    data = load_data(args)

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n{'=' * 80}")
        print(f"CONFIGURATION {i + 1}/{len(configs)}")
        print(f"{'=' * 80}")

        result = run_single_config(tt, cfg, data, args)
        results.append(result)
        save_results(results, args.output_dir, tt)

    print_comparison_table(results, tt)
    save_results(results, args.output_dir, tt)

    if tt == "pfptok":
        save_pfptok_nt_per_token(results, args.output_dir)

    print("\n" + "=" * 80)
    print(f"{tt.upper()} Ablation Study Complate")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
