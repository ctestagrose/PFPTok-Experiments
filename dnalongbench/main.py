import argparse
import gzip
import json
import os
import random
import time
import tracemalloc
from functools import partial
from statistics import mean

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast, TrainingArguments

from models.bert_model import BERT
from models.caduceus_model import CaduceusForClassification
from models.hyena_model import HyenaDNAForClassification
from utils.caduceus_tokenizer import (
    build_caduceus_tokenizer,
    load_caduceus_tokenizer,
    save_caduceus_tokenizer,
    tokenize_sequences_caduceus,
    tokenize_sequences_caduceus_eqtl,
)
from utils.data_utils import (
    create_dataset,
    create_dataset_eqtl,
    tokenize_sequences_no_genes,
    tokenize_sequences_no_genes_eqtl,
)
from utils.dataset import collate_fn
from utils.huggingface_utils import CustomTrainer, compute_metrics
from utils.hyena_tokenizer import (
    build_hyena_tokenizer,
    load_hyena_tokenizer,
    save_hyena_tokenizer,
    tokenize_sequences_hyena,
    tokenize_sequences_hyena_eqtl,
)
from utils.pfptok import PFPTok

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cuda_memory_stats():
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    stats = {"cuda_available": True, "gpus": []}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 2
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 2
        peak = torch.cuda.max_memory_allocated(i) / 1024 ** 2
        stats["gpus"].append({
            "device": i,
            "name": torch.cuda.get_device_name(i),
            "allocated": allocated,
            "reserved": reserved,
            "peak": peak,
        })
    return stats


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def load_jsonl_gz(path):
    data = []
    with gzip.open(path, "rt") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a genomic sequence classification model on DNALongBench tasks."
    )
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument("--model_config", type=str, required=True,
                        help="Path to model configuration JSON.")
    parser.add_argument("--task", type=str, default="ETGP",
                        choices=["ETGP", "eQTL"],
                        help="Which DNALongBench task to run.")
    parser.add_argument("--model_type", type=str, default="bert",
                        choices=["bert", "hyena", "caduceus"],
                        help="Model architecture: bert, hyena, or caduceus.")

    # BERT / PFPTok
    parser.add_argument("--pfp_w", type=int, default=100,
                        help="PFPTok window size w (BERT only).")
    parser.add_argument("--pfp_p", type=int, default=4096,
                        help="PFPTok hash modulus p (BERT only).")

    # Caduceus
    parser.add_argument("--caduceus_model_name", type=str,
                        default="kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
                        help="HuggingFace model ID for the Caduceus backbone.")
    parser.add_argument("--caduceus_pooling", type=str, default="mean",
                        choices=["mean", "max", "first", "last"])
    parser.add_argument("--caduceus_max_length", type=int, default=450_000,
                        help="Max input length; 450000 = no truncation on DNALongBench.")
    parser.add_argument("--caduceus_crop", type=str, default="ends",
                        choices=["ends", "center", "left", "right"],
                        help="Cropping strategy when sequence exceeds caduceus_max_length.")
    parser.add_argument("--caduceus_conjoin_eval", action="store_true",
                        help="Post-hoc RC conjoining at eval on -ph checkpoints (2x eval cost).")

    # HyenaDNA
    parser.add_argument("--hyena_model_name", type=str,
                        default="LongSafari/hyenadna-medium-450k-seqlen-hf",
                        help="HuggingFace model ID for the HyenaDNA backbone.")
    parser.add_argument("--hyena_max_length", type=int, default=450_000,
                        help="Max input length for HyenaDNA tokenization.")

    # Shared
    parser.add_argument("--antibiotic", type=str, required=True,
                        help="Label column / task variant (use 'binary_json' for JSON splits).")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Root directory for saving model checkpoints and results.")
    parser.add_argument("--use_json_dataset", action="store_true",
                        help="Use a prebuilt JSON dataset with train/validation/test splits.")
    parser.add_argument("--json_path", type=str, default="",
                        help="Path stem to the JSONL split files.")
    parser.add_argument("--ordered", action="store_true",
                        help="Order tokenizer training sequences (negatives first).")
    parser.add_argument("--build_tokenizer_fullset", action="store_true",
                        help="Build the tokenizer on the full training set only.")
    parser.add_argument("--track_mem_time", action="store_true",
                        help="Track memory and wall-clock time usage.")
    parser.add_argument("--include_val_in_tokenizer", action="store_true",
                        help="Include validation sequences when building the PFP tokenizer.")
    return parser.parse_args()


def setup_tokenizer_and_wrap(tokenizer_manager, full_set_seqs, args, save_path):
    model = args.model_type.lower()

    if model == "bert":
        tokenizer_path = os.path.join(save_path, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            print(f"Loading PFP tokenizer from {tokenizer_path}")
            raw_tokenizer = tokenizer_manager.load_tokenizer(tokenizer_path)
        else:
            print(f"Training PFP tokenizer (w={args.pfp_w}, p={args.pfp_p})...")
            raw_tokenizer = tokenizer_manager.setup_tokenizer(
                full_set_seqs, w=args.pfp_w, p=args.pfp_p
            )
            tokenizer_manager.save_tokenizer(raw_tokenizer, tokenizer_path)
            print(f"Tokenizer saved to {tokenizer_path}")

        special_tokens_list = [
            "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[INTB]", "[INTA]", "[GENE]"
        ]
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw_tokenizer,
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            additional_special_tokens=[
                tok for tok in special_tokens_list
                if tok not in ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
            ],
        )
        vocab_size = tokenizer.vocab_size
        print(f"PFP tokenizer ready. Vocab size: {vocab_size}")

    elif model == "caduceus":
        tokenizer_path = os.path.join(save_path, "caduceus_tokenizer")
        if os.path.isdir(tokenizer_path) and os.listdir(tokenizer_path):
            print(f"Loading Caduceus tokenizer from {tokenizer_path}")
            tokenizer = load_caduceus_tokenizer(tokenizer_path)
        else:
            print("Building Caduceus character tokenizer...")
            tokenizer = build_caduceus_tokenizer(
                model_name=args.caduceus_model_name,
                model_max_length=args.caduceus_max_length,
            )
            save_caduceus_tokenizer(tokenizer, tokenizer_path)
        vocab_size = tokenizer.vocab_size
        print(f"Caduceus tokenizer ready. Vocab size: {vocab_size}")

    else:  # hyena
        tokenizer_path = os.path.join(save_path, "hyena_tokenizer")
        if os.path.isdir(tokenizer_path) and os.listdir(tokenizer_path):
            print(f"Loading HyenaDNA tokenizer from {tokenizer_path}")
            tokenizer = load_hyena_tokenizer(tokenizer_path)
        else:
            print("Building HyenaDNA character tokenizer...")
            tokenizer = build_hyena_tokenizer(model_max_length=args.hyena_max_length)
            save_hyena_tokenizer(tokenizer, tokenizer_path)
        vocab_size = tokenizer.vocab_size
        print(f"HyenaDNA tokenizer ready. Vocab size: {vocab_size}")

    return tokenizer, vocab_size


def _nucleotide_training_args(fold_num, tag, args, save_path):
    return TrainingArguments(
        output_dir=os.path.join(save_path, f"fold_{fold_num + 1}_{tag}_results"),
        logging_dir=os.path.join(save_path, f"fold_{fold_num + 1}_{tag}_logs"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=5e-6,
        save_safetensors=False,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="best",
        logging_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=0,
        remove_unused_columns=False,
        report_to="none",
        gradient_accumulation_steps=getattr(args, "grad_accum", 1),
        fp16=False,
        bf16=True,
    )


def train_fold_logic(
    fold_num,
    train_fold_data,
    val_fold_data,
    model_config,
    vocab_size,
    wrapped_tokenizer,
    target_format,
    args,
    save_path,
    enable_loss_plotting=True,
    enable_early_stopping=False,
):
    print(f"\n===== Starting Fold {fold_num + 1} {args.model_type} =====")

    if args.task != "eQTL":
        train_dataset = create_dataset(train_fold_data, target_format)
        val_dataset = create_dataset(val_fold_data, target_format)
        train_labels = [label for _, label, _, _ in train_fold_data]
    else:
        train_dataset = create_dataset_eqtl(train_fold_data, target_format)
        val_dataset = create_dataset_eqtl(val_fold_data, target_format)
        train_labels = [label for _, _, label, _, _ in train_fold_data]

    model_type = args.model_type.lower()

    if model_type == "bert":
        config = json.load(open(model_config))
        config["num_labels"] = 1 if target_format == "binary" else len(train_labels[0])
        model = BERT(vocab_size, config, paired=(args.task == "eQTL"))

        fold_output_dir = os.path.join(save_path, f"fold_{fold_num + 1}_results")
        fold_logging_dir = os.path.join(save_path, f"fold_{fold_num + 1}_logs")
        fold_plot_dir = os.path.join(save_path, f"fold_{fold_num + 1}_plots")

        set_seed(12345)

        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            logging_dir=fold_logging_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            learning_rate=5e-6,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,
            remove_unused_columns=False,
            report_to="none",
        )

        if enable_loss_plotting:
            print(f"Fold {fold_num + 1} output dir: {fold_output_dir}")
            print(f"Loss plots will be saved to: {fold_plot_dir}")

        tag = "bert"

    else:
        neg_count = sum(1 for lbl in train_labels if lbl == 0)
        pos_count = sum(1 for lbl in train_labels if lbl == 1)
        pos_weight = neg_count / pos_count

        if model_type == "caduceus":
            print(f"Caduceus pos_weight = {pos_weight:.3f} ({neg_count} neg / {pos_count} pos)")
            model = CaduceusForClassification(
                model_name=args.caduceus_model_name,
                num_labels=1,
                pooling=getattr(args, "caduceus_pooling", "mean"),
                freeze_backbone=False,
                pos_weight=pos_weight,
                conjoin_eval=getattr(args, "caduceus_conjoin_eval", False),
                paired=(args.task == "eQTL"),
            )
            print(
                f"Caduceus: checkpoint={args.caduceus_model_name} "
                f"rcps={model.rcps} conjoin_eval={model.conjoin_eval} "
                f"d_model={model.d_model} paired={model.paired}"
            )
            tag = "caduceus"

        else:  # hyena
            print(f"HyenaDNA pos_weight = {pos_weight:.3f} ({neg_count} neg / {pos_count} pos)")
            model = HyenaDNAForClassification(
                model_name=args.hyena_model_name,
                num_labels=1,
                pooling=getattr(args, "hyena_pooling", "mean"),
                freeze_backbone=False,
                pos_weight=pos_weight,
                paired=(args.task == "eQTL"),
            )
            tag = "hyena"

        set_seed(12345)
        training_args = _nucleotide_training_args(fold_num, tag, args, save_path)

    data_collator = partial(
        collate_fn,
        classification_type=target_format,
        MASK_TOKEN=wrapped_tokenizer.mask_token_id,
        PAD_TOKEN=wrapped_tokenizer.pad_token_id,
        VOCAB_SIZE=wrapped_tokenizer.vocab_size,
    )
    compute_metrics_partial = partial(compute_metrics, target_format=target_format)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=wrapped_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_partial,
        target_format=target_format,
        train_labels=train_labels,
    )
    print(f"Starting {tag.upper()} training for Fold {fold_num + 1}...")

    train_result = trainer.train()
    print(f"Training finished for Fold {fold_num + 1}.")

    if enable_loss_plotting:
        print("\n--- Training Summary ---")
        print(f"Total training steps: {train_result.global_step}")
        print(f"Final training loss:  {train_result.training_loss:.4f}")
        if hasattr(trainer.state, "best_metric"):
            print(f"Best validation metric: {trainer.state.best_metric:.4f}")

    best_checkpoint = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric
    print(f"\n--- Fold {fold_num + 1} Best Model ---")
    print(f"Best checkpoint saved at: {best_checkpoint}")
    print(f"Best validation F1 score: {best_metric:.4f}")

    print(f"Evaluating best model for Fold {fold_num + 1} on validation set...")
    final_eval_metrics = trainer.evaluate(eval_dataset=val_dataset)

    print(f"\n--- Best Validation Metrics (Fold {fold_num + 1}) ---")
    print(f"  Accuracy:       {final_eval_metrics.get('eval_accuracy', 'N/A'):.4f}")
    print(f"  F1 Score:       {final_eval_metrics.get('eval_f1', 'N/A'):.4f}")
    print(f"  Precision:      {final_eval_metrics.get('eval_precision', 'N/A'):.4f}")
    print(f"  Recall:         {final_eval_metrics.get('eval_recall', 'N/A'):.4f}")
    print(f"  AUC:            {final_eval_metrics.get('eval_auc', 'N/A'):.4f}")
    print(f"  Best Threshold: {final_eval_metrics.get('eval_best_threshold', 'N/A')}")
    print(f"  Eval Loss:      {final_eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print("-" * 30)

    return trainer, final_eval_metrics


def _tokenize_split(raw, model_type, wrapped_tokenizer, pfp_tokenizer, args, task):
    if not raw:
        return (None, (0, 0)) if task == "eQTL" else ([], (0, 0))

    if task != "eQTL":
        if model_type == "caduceus":
            _kw = dict(max_length=args.caduceus_max_length, crop=args.caduceus_crop)
            return tokenize_sequences_caduceus(raw, wrapped_tokenizer, **_kw)
        if model_type == "hyena":
            return tokenize_sequences_hyena(raw, wrapped_tokenizer, args.hyena_max_length)
        return tokenize_sequences_no_genes(raw, pfp_tokenizer, wrapped_tokenizer, args)

    if model_type == "caduceus":
        _kw = dict(max_length=args.caduceus_max_length, crop=args.caduceus_crop)
        return tokenize_sequences_caduceus_eqtl(raw, wrapped_tokenizer, **_kw)
    if model_type == "hyena":
        return tokenize_sequences_hyena_eqtl(raw, wrapped_tokenizer, args.hyena_max_length)
    return tokenize_sequences_no_genes_eqtl(raw, pfp_tokenizer, wrapped_tokenizer, args)


def main():
    args = parse_arguments()

    if args.track_mem_time:
        tracemalloc.start()
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

    task = args.task
    model_type = args.model_type.lower()

    if args.use_json_dataset:
        args.antibiotic = "binary_json"

        train_temp = load_jsonl_gz(args.json_path + "_train.jsonl.gz")
        val_temp = load_jsonl_gz(args.json_path + "_valid.jsonl.gz")
        test_temp = load_jsonl_gz(args.json_path + "_test.jsonl.gz")

        print(train_temp[0].keys())

        order_dict = {"quarter": [], "half": [], "three_quarter": [], "full": []}

        train_raw = []
        val_raw = []
        test_raw = []
        all_raw_seqs = []

        positive = [item for item in train_temp if item["label"] == 1]
        negative = [item for item in train_temp if item["label"] == 0]

        if args.ordered:
            print("Using ordered data (negatives first).")
            train_temp = negative + positive

        print(f"Number of negative: {len(negative)}")
        print(f"Number of positive: {len(positive)}")
        print(
            f"Percentage positive: "
            f"{len(positive) / (len(positive) + len(negative)) * 100:.1f}%"
        )

        n_train = len(train_temp)
        for item in train_temp:
            seq = item["seq_ref"] if task == "eQTL" else item["seq"]
            seq_2 = item.get("seq_alt") if task == "eQTL" else None
            if len(order_dict["quarter"]) <= 0.25 * n_train:
                order_dict["quarter"].append(seq)
            if len(order_dict["half"]) <= 0.5 * n_train:
                order_dict["half"].append(seq)
            if len(order_dict["three_quarter"]) <= 0.75 * n_train:
                order_dict["three_quarter"].append(seq)
            all_raw_seqs.append(seq)
            if task == "eQTL" and seq_2:
                all_raw_seqs.append(seq_2)

        if args.include_val_in_tokenizer:
            print("Including validation sequences in tokenizer training.")
            n_total = n_train + len(val_temp)
            for item in val_temp:
                seq = item["seq_ref"] if task == "eQTL" else item["seq"]
                seq_2 = item.get("seq_alt") if task == "eQTL" else None
                if len(order_dict["quarter"]) <= 0.25 * n_total:
                    order_dict["quarter"].append(seq)
                if len(order_dict["half"]) <= 0.5 * n_total:
                    order_dict["half"].append(seq)
                if len(order_dict["three_quarter"]) <= 0.75 * n_total:
                    order_dict["three_quarter"].append(seq)
                all_raw_seqs.append(seq)
                if task == "eQTL" and seq_2:
                    all_raw_seqs.append(seq_2)

        order_dict["full"] = all_raw_seqs

        if task == "eQTL":
            for item in train_temp:
                train_raw.append((item["seq_ref"], item["seq_alt"], item["label"], "placehold"))
            for item in val_temp:
                val_raw.append((item["seq_ref"], item["seq_alt"], item["label"], "placehold"))
            for item in test_temp:
                test_raw.append((item["seq_ref"], item["seq_alt"], item["label"], "placehold"))
        else:
            for item in train_temp:
                train_raw.append((item["seq"], item["label"], "placehold"))
            for item in val_temp:
                val_raw.append((item["seq"], item["label"], "placehold"))
            for item in test_temp:
                test_raw.append((item["seq"], item["label"], "placehold"))

        amounts = ["full"] if args.build_tokenizer_fullset else [
            "quarter", "half", "three_quarter", "full"
        ]
        print(f"Will build tokenizer and train model for: {amounts}")

        for amount in amounts:
            save_path = os.path.join(args.save_path, args.antibiotic, amount)
            os.makedirs(save_path, exist_ok=True)

            pfp_tokenizer = PFPTok()
            wrapped_tokenizer, vocab_size = setup_tokenizer_and_wrap(
                tokenizer_manager=pfp_tokenizer,
                full_set_seqs=order_dict[amount],
                args=args,
                save_path=save_path,
            )

            tr_tok, tr_counts = _tokenize_split(
                train_raw, model_type, wrapped_tokenizer, pfp_tokenizer, args, task
            )
            va_tok, va_counts = _tokenize_split(
                val_raw, model_type, wrapped_tokenizer, pfp_tokenizer, args, task
            )
            te_tok, te_counts = _tokenize_split(
                test_raw, model_type, wrapped_tokenizer, pfp_tokenizer, args, task
            )

            with open(os.path.join(save_path, "Sequence_Information.txt"), "w") as f:
                f.write(f"TRAIN UNK/NONUNK: {tr_counts}\n")
                f.write(f"VAL   UNK/NONUNK: {va_counts}\n")
                f.write(f"TEST  UNK/NONUNK: {te_counts}\n")
                f.write(f"VOCAB SIZE: {vocab_size}\n")
                f.write(f"Number of negative: {len(negative)}\n")
                f.write(f"Number of positive: {len(positive)}\n")
                f.write(
                    f"Percentage positive: "
                    f"{len(positive) / (len(positive) + len(negative)) * 100:.1f}%\n"
                )

            target_format = "binary"

            if task != "eQTL":
                test_dataset = create_dataset(te_tok, target_format) if te_tok else None
            else:
                test_dataset = create_dataset_eqtl(te_tok, target_format) if te_tok else None

            pos = sum(1 for item in tr_tok if item[1] == 1)
            neg = sum(1 for item in tr_tok if item[1] == 0)
            print(f"Train split — positive: {pos}, negative: {neg}")

            train_len = [len(item[0][0]) for item in tr_tok]
            valid_len = [len(item[0][0]) for item in va_tok]
            test_len = [len(item[0][0]) for item in te_tok]

            print(
                f"Max seq lengths — train: {max(train_len)}, "
                f"val: {max(valid_len)}, test: {max(test_len)}"
            )
            print(
                f"Mean seq lengths — train: {mean(train_len):.0f}, "
                f"val: {mean(valid_len):.0f}, test: {mean(test_len):.0f}"
            )

            trainer, eval_metrics = train_fold_logic(
                fold_num=0,
                train_fold_data=tr_tok,
                val_fold_data=va_tok,
                model_config=args.model_config,
                vocab_size=vocab_size,
                wrapped_tokenizer=wrapped_tokenizer,
                target_format=target_format,
                args=args,
                save_path=save_path,
                enable_loss_plotting=True,
                enable_early_stopping=False,
            )

            if test_dataset is not None:
                print("\n--- Evaluating best model on JSON test split ---")
                try:
                    trainer._load_best_model()
                except Exception:
                    pass
                test_results = trainer.predict(test_dataset=test_dataset)
                with open(os.path.join(save_path, "test_results.json"), "w") as fj:
                    json.dump(to_jsonable(test_results.metrics), fj, indent=2)
                print(json.dumps(
                    {k: float(v) if hasattr(v, "item") else v
                     for k, v in test_results.metrics.items()},
                    indent=2,
                ))

            print("\n[JSON mode finished]")

        if args.track_mem_time:
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            elapsed = time.time() - t0
            tracemalloc.stop()
            cuda_stats = get_cuda_memory_stats()

            final_save = os.path.join(args.save_path, args.antibiotic)
            with open(os.path.join(final_save, "Memory_and_Time.txt"), "w") as f:
                f.write(f"Elapsed time        : {elapsed:.1f} s  ({elapsed / 60:.1f} min)\n")
                f.write("\n--- CPU Memory (tracemalloc) ---\n")
                f.write(f"Current             : {current_mem / 1024 ** 2:.1f} MB\n")
                f.write(f"Peak                : {peak_mem / 1024 ** 2:.1f} MB\n")
                f.write("\n--- CUDA Memory ---\n")
                if not cuda_stats["cuda_available"]:
                    f.write("CUDA not available\n")
                else:
                    for gpu in cuda_stats["gpus"]:
                        f.write(f"GPU {gpu['device']} ({gpu['name']})\n")
                        f.write(f"  Allocated (current) : {gpu['allocated']:.1f} MB\n")
                        f.write(f"  Reserved  (current) : {gpu['reserved']:.1f} MB\n")
                        f.write(f"  Peak allocated      : {gpu['peak']:.1f} MB\n")


if __name__ == "__main__":
    main()
