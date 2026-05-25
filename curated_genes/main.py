import argparse
import json
import glob
import random
from collections import defaultdict
import torch
import numpy as np
from functools import partial
from transformers import TrainingArguments, PreTrainedTokenizerFast
from data_prep import DataPreparer
from utils.gene_manager import GeneManager
from utils.sequence_processor import SequenceProcessor
from tokenizers.pfp_tokenizer import TokenizerManager
from utils.dataset import collate_fn
from models.bert_model import BERT
from utils.huggingface_utils import CustomTrainer, compute_metrics, attention_token_importance
from utils.data_utils import create_dataset, tokenize_sets, create_folds
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_arguments():
    with open('./Config/train_config.json', 'r') as f:
        file_args = json.load(f)

    parser = argparse.ArgumentParser(description='Train a BERT model for MTB Antibiotic Resistance.')

    parser.add_argument('--sequence_dir', type=str, required=True, help='Path to the directory of Fasta Files.')
    parser.add_argument('--test_sequence_dir', type=str, required=True, help='Path to the directory of Fasta Files.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--stride', type=int, default=1, help='Stride for Kmer creation')
    parser.add_argument('--Kmer_Size', type=int, default=31, help='The size of the Kmers (31-default)')
    parser.add_argument('--model_config', type=str, required=True, help='Path to Model configuration')
    parser.add_argument('--antibiotic', type=str, required=True, help='All, Rare only, or provide one antibiotic abbrev.')
    parser.add_argument('--save_path', type=str, required=True, help='Where to save model?')
    parser.add_argument('--use_holdout', action='store_true', help='Use a hold out test set?')
    parser.add_argument('--use_gene_file', action='store_true', help='Use all the genes in a fasta or the gene file?')

    parser.add_argument('--use_scaffolds', action='store_true',
                        help='Treat each FASTA record as a scaffold and skip per-gene filtering.')
    parser.add_argument('--scaffold_concat', action='store_true',
                        help='When used with --use_scaffolds, concatenate all scaffolds per isolate into a single sequence.')
    parser.add_argument('--scaffold_filename', type=str, default='scaffolds.fasta',
                        help='Name of the FASTA file inside each isolate directory (default: scaffolds.fasta).')
    parser.add_argument('--isolate_id_from_dir', action='store_true',
                        help='When reading scaffolds.fasta inside isolate directories, use the parent directory name as the isolate id.')
    parser.add_argument('--gene_file', type=str, required=True, help='Path to the genes JSON file.')
    parser.add_argument('--target_file', type=str, required=True, help='Path to the targets file.')
    parser.add_argument('--p_base', type=float, default=0.15,
                   help='Base probability of k-mer fallback for known phrases (default: 0.15)')
    parser.add_argument('--fallback_strategy', type=str, default='proportional',
                       choices=['proportional', 'inverse'],
                       help='Strategy for computing fallback probability based on frequency')
    parser.add_argument('--fallback_alpha', type=float, default=0.7,
                       help='Alpha parameter for fallback probability scaling (default: 0.7)')
    parser.add_argument('--min_count_uncommon', type=int, default=5,
                       help='Minimum count for a phrase to be considered common (default: 2)')
    parser.add_argument('--rare_quantile', type=float, default=0.30,
                       help='Quantile threshold for rare phrases (default: 0.20)')

    # Tokenizer selection
    parser.add_argument('--tokenizer_type', type=str, default='pfp',
                        choices=['pfp', 'bpe', 'unigram'],
                        help='Type of tokenizer: pfp, bpe, or unigram')

    # PFP tokenizer parameters
    parser.add_argument('--pfp_w', type=int, default=30, help='PFP window size')
    parser.add_argument('--pfp_d', type=int, default=227, help='PFP hash divisor')

    # BPE/Unigram tokenizer parameters
    parser.add_argument('--bpe_vocab_size', type=int, default=100000,
                        help='Vocabulary size for BPE/Unigram tokenizer')
    parser.add_argument('--bpe_min_frequency', type=int, default=5,
                        help='Minimum frequency for BPE merges')
    parser.add_argument('--bpe_dropout', type=float, default=0.0,
                        help='BPE dropout probability during training')
    parser.add_argument('--disable_pfp_fallback', action='store_true',
                        help='Disable k-mer fallback for PFP tokenizer')
    parser.add_argument('--scaffold_spacer_ns', type=int, default=0,
                        help='Number of Ns to insert between scaffolds during concatenation.')

    # Seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    argument_list = []
    for key, value in file_args.items():
        if isinstance(value, bool):
            if value:
                argument_list.append(f'--{key}')
        else:
            argument_list.append(f'--{key}')
            argument_list.append(str(value))

    args = parser.parse_args(argument_list)
    return args


def get_isolate_scaffold_files(sequence_dir, scaffold_filename="scaffolds.fasta"):
    """Return list of <sequence_dir>/<ISOLATE>/<isolate>.H37Rv_ordered.fasta that actually exist."""
    files = []
    for isolate in os.listdir(sequence_dir):
        iso_dir = os.path.join(sequence_dir, isolate)
        if not os.path.isdir(iso_dir):
            continue
        f = os.path.join(iso_dir, isolate + ".H37Rv_ordered.fasta")
        if os.path.exists(f):
            files.append(f)
    print(f"Found {len(files)} isolate scaffold files.")
    return files


def setup_and_load_data(args):
    save_path = os.path.join(args.save_path, args.antibiotic)
    os.makedirs(save_path, exist_ok=True)

    print("Starting to get Fastas")

    if args.use_scaffolds:
        fasta_files = get_isolate_scaffold_files(args.sequence_dir,
                                                 getattr(args, "scaffold_filename", "scaffolds.fasta"))
    else:
        fasta_files = glob.glob(os.path.join(args.sequence_dir, "**", "*.fasta"), recursive=True)

    random.Random(17).shuffle(fasta_files)
    full_set = fasta_files

    gene_manager = GeneManager(args.gene_file)
    sequence_processor = SequenceProcessor(args.Kmer_Size, args.stride)
    if args.tokenizer_type == 'bpe':
        from tokenizers.bpe_tokenizer import TokenizerManagerBPE
        tokenizer_manager = TokenizerManagerBPE(vocab_size=args.bpe_vocab_size)
    elif args.tokenizer_type == 'unigram':
        from tokenizers.unigram_tokenizer import TokenizerManagerUnigram
        tokenizer_manager = TokenizerManagerUnigram(vocab_size=args.bpe_vocab_size)
    else:
        tokenizer_manager = TokenizerManager()
    data_preparer = DataPreparer(gene_manager, sequence_processor, tokenizer_manager, args)

    print("Preparing data...")
    zipped_data, full_set_seqs = data_preparer.prep_data(
        fasta_files=fasta_files,
        full_set=full_set,
        target_file=args.target_file,
        target_format=args.antibiotic,
        mode="Train"
    )
    print(f"Loaded {len(zipped_data)} samples.")

    return zipped_data, full_set_seqs, sequence_processor, tokenizer_manager, save_path


def load_test_data(args, gene_manager, sequence_processor, tokenizer_manager):
    if not args.test_sequence_dir:
        print("No test sequence directory provided, skipping test set loading.")
        return None, None, None, None

    print(f"Loading test data from: {args.test_sequence_dir}")
    if args.use_scaffolds:
        test_fasta_files = get_isolate_scaffold_files(args.test_sequence_dir,
                                                 getattr(args, "scaffold_filename", "scaffolds.fasta"))
    else:
        test_fasta_files = glob.glob(os.path.join(args.test_sequence_dir, "**", "*.fasta"), recursive=True)

    if not test_fasta_files:
        print(f"Warning: No FASTA files found in {args.test_sequence_dir}")
        return None, None, None, None

    data_preparer_test = DataPreparer(gene_manager, sequence_processor, tokenizer_manager, args)

    zipped_test_data, _ = data_preparer_test.prep_data(
        fasta_files=test_fasta_files,
        full_set=[],
        target_file=args.target_file,
        target_format=args.antibiotic,
        mode="Evaluate"
    )

    if not zipped_test_data:
        print("Warning: No valid test samples prepared.")
        return None, None, None, None

    print(f"Loaded {len(zipped_test_data)} test samples.")

    test_sequences = [seq for seq, _, _, _ in zipped_test_data]
    test_labels_raw = [label for _, label, _, _ in zipped_test_data]
    test_seq_ids = [seq_id for _, _, seq_id, _ in zipped_test_data]
    test_genes_list = [genes for _, _, _, genes in zipped_test_data]

    _, test_prepped_seqs, test_prepped_labels = sequence_processor.extract_and_prep_genes(test_sequences, test_labels_raw)

    return test_prepped_seqs, test_prepped_labels, test_seq_ids, test_genes_list


# Tokenization and Wrapping
def setup_tokenizer_and_wrap(tokenizer_manager, full_set_seqs, w, d, args, save_path):
    sequence_processor = SequenceProcessor(args.Kmer_Size, args.stride)
    print(f"Setting up {args.tokenizer_type.upper()} tokenizer...")

    if args.tokenizer_type == 'bpe':
        tokenizer_filename = "tokenizer_bpe.json"
    elif args.tokenizer_type == 'unigram':
        tokenizer_filename = "tokenizer_unigram.json"
    else:
        tokenizer_filename = "tokenizer.json"

    tokenizer_path = os.path.join(save_path, tokenizer_filename)

    if os.path.exists(tokenizer_path) or os.path.exists(os.path.dirname(tokenizer_path) + "/tokenizer_config.json"):
        print(f"Loading tokenizer from {tokenizer_path}")
        wrapped_tokenizer = tokenizer_manager.load_tokenizer(tokenizer_path)
    else:
        print(f"Training {args.tokenizer_type.upper()} tokenizer...")

        if args.tokenizer_type == 'bpe':
            wrapped_tokenizer = tokenizer_manager.setup_tokenizer(
                full_set_seqs,
                vocab_size=args.bpe_vocab_size,
                min_frequency=args.bpe_min_frequency,
            )
        elif args.tokenizer_type == 'unigram':
            wrapped_tokenizer = tokenizer_manager.setup_tokenizer(
                full_set_seqs,
                vocab_size=args.bpe_vocab_size,
            )
        else:
            # PFP tokenizer
            tokenizer = tokenizer_manager.setup_tokenizer(
                full_set_seqs, w, d,
                min_count_uncommon=getattr(args, 'min_count_uncommon', 2),
                rare_quantile=getattr(args, 'rare_quantile', 0.20)
            )

            special_tokens_list = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[INTB]", "[INTA]", "[GENE]"]
            wrapped_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                bos_token="[CLS]", eos_token="[SEP]", unk_token="[UNK]",
                sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                mask_token="[MASK]",
                additional_special_tokens=[tok for tok in special_tokens_list if
                                           tok not in ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]]
            )

    vocab_size = wrapped_tokenizer.vocab_size
    print(f"Final Vocab Size: {vocab_size}")

    return wrapped_tokenizer, vocab_size, tokenizer_manager


def _as_seq_list(x):
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        if len(x) and isinstance(x[0], torch.Tensor):
            return [t.detach().cpu().tolist() for t in x]
        return x
    raise TypeError(f"Unsupported type for gene_ids: {type(x)}")


def collect_top_tokens_and_genes(
    model,
    dataloader,
    gene_pad_id=-100,
    normalize=True,
    use_cls=True,
    topk=10,
    save_all_tokens=True,
    tokenizer=None):

    all_samples = []
    global_gene_counter = defaultdict(float)
    global_token_counter = defaultdict(float)

    for batch_idx, batch in enumerate(dataloader):
        tok_scores, gene_scores = attention_token_importance(
            model,
            batch,
            gene_pad_id=gene_pad_id,
            normalize=normalize,
            use_cls=use_cls,
        )

        gids_all = _as_seq_list(batch["gene_ids"])
        input_ids_all = _as_seq_list(batch["input_ids"])
        B = len(tok_scores)

        for b in range(B):
            valid = int(batch["attention_mask"][b].sum().item())
            gids_row = gids_all[b]
            token_ids_row = input_ids_all[b]

            if len(gids_row) < valid:
                gids = gids_row + [gids_row[-1]] * (valid - len(gids_row))
            else:
                gids = gids_row[:valid]

            token_ids = token_ids_row[:valid]

            full_sequence = ""
            if tokenizer is not None:
                full_sequence = tokenizer.decode(token_ids, skip_special_tokens=True)

            token_sequences = []
            if tokenizer is not None:
                for tid in token_ids:
                    token_sequences.append(tokenizer.decode([tid]))

            ts = tok_scores[b]

            if save_all_tokens:
                sample_data = {
                    "sample_id": f"batch{batch_idx}_item{b}",
                    "sequence_length": valid,
                    "full_sequence": full_sequence,
                    "token_ids": token_ids,
                    "token_sequences": token_sequences,
                    "token_scores": ts.tolist(),
                    "gene_ids": gids,
                    "top_k_indices": ts.topk(min(topk, ts.numel())).indices.tolist(),
                }
            else:
                top_idx = ts.topk(min(topk, ts.numel())).indices.tolist()
                sample_data = {
                    "sample_id": f"batch{batch_idx}_item{b}",
                    "full_sequence": full_sequence,
                    "top_tokens": top_idx,
                    "top_token_sequences": [token_sequences[j] for j in top_idx] if token_sequences else [],
                    "top_genes": [gids[j] for j in top_idx],
                    "token_scores": ts.tolist(),
                }

            label = None
            species = None
            if "labels" in batch:
                label = int(batch["labels"][b].item())
            if "binary" in batch:
                label = int(batch["binary"][b].item())
            if "species" in batch and isinstance(batch["species"], list):
                species = batch["species"][b]

            sample_data.update({
                "label": label,
                "species": species or "",
            })

            all_samples.append(sample_data)

            for g, score in gene_scores[b].items():
                global_gene_counter[g] += score

            for j, (token_id, gene_id, score) in enumerate(zip(token_ids, gids, ts.tolist())):
                global_token_counter[(gene_id, token_id, j)] += score

    sorted_genes = sorted(global_gene_counter.items(), key=lambda kv: kv[1], reverse=True)
    sorted_tokens = sorted(global_token_counter.items(), key=lambda kv: kv[1], reverse=True)

    return all_samples, sorted_genes, sorted_tokens


# Training Function for a Single Fold
def train_fold_logic(fold_num, train_fold_data, val_fold_data, model_config, vocab_size,
                     wrapped_tokenizer, target_format, antibiotic, args, save_path):
    print(f"\n===== Starting Fold {fold_num + 1} =====")
    train_dataset = create_dataset(train_fold_data, target_format)
    val_dataset = create_dataset(val_fold_data, target_format)
    train_labels_for_loss = [label for _, label, _, _ in train_fold_data]

    config = json.load(open(model_config))
    config['num_labels'] = 1 if target_format == 'binary' else len(train_labels_for_loss[0])
    model = BERT(vocab_size, config)

    fold_output_dir = os.path.join(save_path, f'fold_{fold_num + 1}_results')
    fold_logging_dir = os.path.join(save_path, f'fold_{fold_num + 1}_logs')

    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        logging_dir=fold_logging_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=config.get('learning_rate', 5e-5),
        lr_scheduler_type="cosine_with_restarts",
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none"
    )

    data_collator = partial(collate_fn,
                            classification_type=target_format,
                            MASK_TOKEN=wrapped_tokenizer.mask_token_id,
                            PAD_TOKEN=wrapped_tokenizer.pad_token_id,
                            VOCAB_SIZE=vocab_size)

    compute_metrics_partial = partial(compute_metrics, target_format=target_format)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=wrapped_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_partial,
        target_format=target_format,
        train_labels=train_labels_for_loss
    )

    print(f"Starting training for Fold {fold_num + 1}...")
    trainer.train()
    print(f"Training finished for Fold {fold_num + 1}.")

    best_checkpoint = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric
    print(f"\n--- Fold {fold_num + 1} Best Model ---")
    print(f"Best checkpoint saved at: {best_checkpoint}")
    print(f"Best validation F1 score: {best_metric:.4f}")

    print(f"Evaluating best model for Fold {fold_num + 1} on Validation Set...")
    final_eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
    test_dl = trainer.get_test_dataloader(val_dataset)

    all_samples, sorted_genes, sorted_tokens = collect_top_tokens_and_genes(
        trainer.model,
        test_dl,
        gene_pad_id=-100,
        normalize=True,
        use_cls=True,
        topk=10,
        save_all_tokens=True,
        tokenizer=wrapped_tokenizer
    )

    with open(os.path.join(args.save_path, antibiotic, f"fold{fold_num+1}_token_gene_analysis_validation.json"), "w") as f:
        json.dump({
            "samples": all_samples,
            "sorted_genes": sorted_genes,
            "sorted_tokens": sorted_tokens,
        }, f, indent=2)

    print("\n=== Top genes across validation dataset ===")
    for g, score in sorted_genes[:20]:
        print(f"{g}: {score:.4f}")

    print("\n=== Top (gene, token_idx) validation pairs ===")
    for (g, token_id, idx), score in sorted_tokens[:20]:
        print(f"{g} @ token {token_id} (pos {idx}): {score:.4f}")

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


# Main
def main():
    args = parse_arguments()

    # Define antibiotics list based on args.antibiotic
    antibiotic_lower = args.antibiotic.lower()
    if antibiotic_lower == "all" or antibiotic_lower == "all genes":
        antibiotics = ["AMI", "ETH", "INH", "RIF", "LEV", "EMB", "RFB", "MXF", "KAN", "LZD", "BDQ", "DLM", "CFZ"]
    elif antibiotic_lower in ("rare", "rare_genes_drugs"):
        antibiotics = ["LZD", "BDQ", "DLM", "CFZ"]
    elif antibiotic_lower == "non-rare":
        antibiotics = ["ETH", "INH", "RIF", "LEV", "EMB", "RFB", "MXF", "KAN"]
    elif antibiotic_lower.startswith("multi-cat"):
        antibiotics = [args.antibiotic]
    else:
        antibiotics = [args.antibiotic]

    overall_results = {}

    for antibiotic in antibiotics:
        print(f"\n{'=' * 10} Processing Antibiotic: {antibiotic} {'=' * 10}")
        w = args.pfp_w
        d = args.pfp_d
        args.antibiotic = antibiotic

        # 1. Setup, Load Train/Val Data
        zipped_data, full_set_seqs, sequence_processor, tokenizer_manager, save_path = setup_and_load_data(args)
        gene_manager = GeneManager(args.gene_file)

        print(f"Length of zipped_data: {len(zipped_data)}")
        print(f"Length of full_set_seqs: {len(full_set_seqs)}")

        # 2. Load Test Data
        test_prepped_seqs, test_prepped_labels, test_seq_ids, test_genes_list = load_test_data(
            args, gene_manager, sequence_processor, tokenizer_manager
        )
        has_test_data = test_prepped_seqs is not None
        
        print(f"Running on Test Data after training: {has_test_data}")

        # 3. Setup Tokenizer
        wrapped_tokenizer, vocab_size, tokenizer_manager = setup_tokenizer_and_wrap(
            tokenizer_manager, full_set_seqs, w, d, args, save_path
        )

        if has_test_data:
            print(f"Length of test_prepped_seqs: {len(test_prepped_seqs)}")

        # 4. Prepare & Tokenize Train/Val Data
        target_format = "binary" if "multi-cat" not in args.antibiotic else "multi-cat"
        all_labels_raw = [label for _, label, _, _ in zipped_data]
        all_sequences = [seq for seq, _, _, _ in zipped_data]
        all_seq_ids = [seq_id for _, _, seq_id, _ in zipped_data]
        all_genes_list = [genes for _, _, _, genes in zipped_data]
        _, all_prepped_seqs, all_prepped_labels = sequence_processor.extract_and_prep_genes(
            all_sequences, all_labels_raw
        )

        tokenizer_for_encoding = wrapped_tokenizer

        all_tokenized_data, unk_counts_tr = tokenize_sets(
            tokenizer_manager,
            all_prepped_seqs,
            all_prepped_labels,
            all_seq_ids,
            all_genes_list,
            tokenizer_for_encoding,
            "train",
            args
        )

        print(f"Tokenized train/val samples: {len(all_tokenized_data)}")

        # 5. Tokenize Test Data
        tokenized_test_data = None
        if has_test_data:
            print("Tokenizing test data...")
            tokenized_test_data, unk_counts_te = tokenize_sets(
                tokenizer_manager,
                test_prepped_seqs,
                test_prepped_labels,
                test_seq_ids,
                test_genes_list,
                tokenizer_for_encoding,
                "test",
                args
            )
            print(f"Tokenized {len(tokenized_test_data)} test samples.")

        # Compute sequence length statistics
        average_len = sum(len(item[0][0]) for item in all_tokenized_data)
        _max_len = max(len(item[0][0]) for item in all_tokenized_data)
        print(f"AVERAGE LENGTH OF TOKENIZED TRAIN/VAL SEQ: {average_len / len(all_tokenized_data)}")

        if tokenized_test_data:
            average_test_len = sum(len(item[0][0]) for item in tokenized_test_data)
            _max_len_test = max(len(item[0][0]) for item in tokenized_test_data)
            print(f"AVERAGE LENGTH OF TOKENIZED TEST SEQ: {average_test_len / len(tokenized_test_data)}")
            print(f"Max lengths - Train: {_max_len}, Test: {_max_len_test}")

        # Write sequence metrics
        with open(os.path.join(args.save_path, antibiotic, "Sequence_Metrics.txt"), 'w') as f:
            f.write(f"AVERAGE LENGTH OF TOKENIZED TRAIN/VAL SEQ: {average_len / len(all_tokenized_data)}\n")
            if tokenized_test_data:
                f.write(f"AVERAGE LENGTH OF TOKENIZED TEST SEQ: {average_test_len / len(tokenized_test_data)}\n")
                f.write(f"UNK COUNT: {unk_counts_te[0]}\n")
                f.write(f"NON UNK COUNT: {unk_counts_te[1]}\n")
            f.write(f"VOCAB SIZE: {vocab_size}")

        # 6. Create Test Dataset
        test_dataset = None
        if has_test_data and tokenized_test_data:
            test_dataset = create_dataset(tokenized_test_data, target_format)
            print("Test dataset created.")

        # 7. Split Train/Val Data for Folds
        train_val_data = all_tokenized_data
        labels_for_fold_split = [lbl for _, lbl, _, _ in train_val_data]
        folds = create_folds(train_val_data, labels_for_fold_split, target_format, n_splits=5, val_size=0.20)

        fold_metrics_val = []
        fold_metrics_test = []

        # 8. Run Training Folds
        for i, (train_fold, val_fold) in enumerate(folds):
            trainer, eval_metrics = train_fold_logic(
                fold_num=i,
                train_fold_data=train_fold,
                val_fold_data=val_fold,
                model_config=args.model_config,
                vocab_size=vocab_size,
                wrapped_tokenizer=wrapped_tokenizer,
                target_format=target_format,
                antibiotic=antibiotic,
                args=args,
                save_path=save_path
            )
            fold_metrics_val.append(eval_metrics)

            # Evaluate on Test Set after fold training
            if test_dataset:
                print(f"\n--- Evaluating Best Model from Fold {i + 1} on Separate Test Set ---")
                best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
                if best_ckpt is not None:
                    try:
                        trainer._load_best_model()
                        if int(os.environ.get("RANK", "0")) == 0:
                            print(f"[INFO] Loaded best model from: {best_ckpt}")
                    except Exception:
                        pt_path = os.path.join(best_ckpt, "pytorch_model.bin")
                        if os.path.exists(pt_path):
                            state = torch.load(pt_path, map_location=trainer.args.device)
                            trainer.model.load_state_dict(state, strict=True)
                            if int(os.environ.get("RANK", "0")) == 0:
                                print(f"[INFO] Loaded best model state_dict from: {pt_path}")
                        else:
                            if int(os.environ.get("RANK", "0")) == 0:
                                print("[WARN] Could not auto-load best checkpoint; proceeding with current weights.")
                else:
                    if int(os.environ.get("RANK", "0")) == 0:
                        print("[WARN] No best_model_checkpoint recorded; proceeding with current weights.")

                test_results = trainer.predict(test_dataset=test_dataset)
                print(f"\n--- Test Set Metrics (Fold {i + 1}) ---")
                print(f"  Test Accuracy:  {test_results.metrics.get('test_accuracy', 'N/A'):.4f}")
                print(f"  Test F1 Score:  {test_results.metrics.get('test_f1', 'N/A'):.4f}")
                print(f"  Test Precision: {test_results.metrics.get('test_precision', 'N/A'):.4f}")
                print(f"  Test Recall:    {test_results.metrics.get('test_recall', 'N/A'):.4f}")
                print(f"  Test AUC:       {test_results.metrics.get('test_auc', 'N/A'):.4f}")
                print(f"  Test Best Threshold Found: {test_results.metrics.get('test_best_threshold', 'N/A')}")
                print(f"  Test Loss:      {test_results.metrics.get('test_loss', 'N/A'):.4f}")
                print("-" * 30)

                test_dl = trainer.get_test_dataloader(test_dataset)
                all_samples, sorted_genes, sorted_tokens = collect_top_tokens_and_genes(
                    trainer.model, test_dl,
                    gene_pad_id=-100, normalize=True, use_cls=True,
                    topk=10, save_all_tokens=True, tokenizer=wrapped_tokenizer
                )

                with open(os.path.join(args.save_path, antibiotic, f"fold_{i + 1}_token_gene_analysis_test.json"), "w") as f:
                    json.dump({
                        "samples": all_samples,
                        "sorted_genes": sorted_genes,
                        "sorted_tokens": sorted_tokens,
                    }, f, indent=2)

                print("\n=== Top genes across test dataset ===")
                for g, score in sorted_genes[:20]:
                    print(f"{g}: {score:.4f}")

                print("\n=== Top (gene, token_idx) test pairs ===")
                for (g, token_id, idx), score in sorted_tokens[:20]:
                    print(f"{g} @ token {token_id} (pos {idx}): {score:.4f}")

                fold_metrics_test.append(test_results.metrics)
                with open(os.path.join(args.save_path, antibiotic, f"test_result_fold_{i+1}.json"), 'w') as f:
                    json.dump(fold_metrics_test, f, indent=2)

        # 9. Aggregate Fold Results
        print(f"\n{'=' * 10} Aggregate Results for: {antibiotic} {'=' * 10}")
        avg_val_f1 = np.mean([m.get('eval_f1', 0) for m in fold_metrics_val])
        avg_val_acc = np.mean([m.get('eval_accuracy', 0) for m in fold_metrics_val])
        print(f"Average Validation F1 across {len(folds)} folds: {avg_val_f1:.4f}")
        print(f"Average Validation Accuracy across {len(folds)} folds: {avg_val_acc:.4f}")
        overall_results[antibiotic] = {'avg_val_f1': avg_val_f1, 'avg_val_acc': avg_val_acc}

        if fold_metrics_test:
            avg_test_f1 = np.mean([m.get('test_f1', 0) for m in fold_metrics_test])
            avg_test_acc = np.mean([m.get('test_accuracy', 0) for m in fold_metrics_test])
            print(f"Average Test F1 across {len(folds)} folds: {avg_test_f1:.4f}")
            print(f"Average Test Accuracy across {len(folds)} folds: {avg_test_acc:.4f}")
            overall_results[antibiotic]['avg_test_f1'] = avg_test_f1
            overall_results[antibiotic]['avg_test_acc'] = avg_test_acc

    # End of all antibiotics
    print("\n===== Overall Summary =====")
    print(json.dumps(overall_results, indent=2))
    with open(os.path.join(args.save_path, "overall_summary.json"), 'w') as f:
        json.dump(overall_results, f, indent=2)


if __name__ == '__main__':
    main()