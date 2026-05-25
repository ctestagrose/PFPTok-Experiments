from .Dataset import SeqDataset
import json
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, train_test_split
from tqdm import tqdm


def create_dataset(zipped_data, classification_type):
    seqs_ = []
    labs = []
    seq_ids = []
    gene_ids = []
    for seqs, label, seq_id, gene_id in zipped_data:
        for seq in seqs:
            seqs_.append(seq)
            labs.append(label)
            seq_ids.append(seq_id)
            gene_ids.append(gene_id)
    return SeqDataset(seqs_, labs, seq_ids, gene_ids, classification_type)


def tokenize_sets(tokenizer_manager, prepped_seqs, prepped_labels, seq_ids, genes_list, tokenizer, split, args):
    encoded_sequences = []
    sequence_genes = []
    gene_mapping = {}
    unk = 0
    non_unk = 0

    tokenizer_type = getattr(args, 'tokenizer_type', 'pfp')

    if tokenizer_type in ('bpe', 'unigram'):
        dropout_prob = getattr(args, 'bpe_dropout', 0.0) if split == "train" else 0.0

        for index, sequence in enumerate(tqdm(prepped_seqs, desc=f"Tokenizing {split} data with {tokenizer_type.upper()}")):
            genes_in_this_isolate = genes_list[index]

            encoded_sequence, token_gene_map, unk_counts = tokenizer_manager.encode_sequences_genes(
                sequence,
                tokenizer,
                genes_in_this_isolate,
                split=split,
                dropout_prob=dropout_prob,
                seed=getattr(args, 'seed', None)
            )

            unk += unk_counts[0]
            non_unk += unk_counts[1]
            encoded_sequences.append([encoded_sequence])
            sequence_genes.append([token_gene_map])

            for token_id, gene in zip(encoded_sequence, token_gene_map):
                if token_id == tokenizer.pad_token_id:
                    continue
                if gene not in gene_mapping:
                    gene_mapping[gene] = []
                if token_id not in gene_mapping[gene]:
                    gene_mapping[gene].append(token_id)

    else:  # PFP tokenizer
        p_base = getattr(args, 'p_base', 0.15)
        strategy = getattr(args, 'fallback_strategy', 'proportional')
        alpha = getattr(args, 'fallback_alpha', 0.7)
        disable_fallback = getattr(args, 'disable_pfp_fallback', False)

        for index, sequence in enumerate(tqdm(prepped_seqs, desc=f"Tokenizing {split} data with PFP")):
            genes_in_this_isolate = genes_list[index]

            encoded_sequence, token_gene_map, unk_counts = tokenizer_manager.encode_sequences_genes(
                sequence,
                tokenizer,
                genes_in_this_isolate,
                seed=getattr(args, 'seed', None)
            )

            unk += unk_counts[0]
            non_unk += unk_counts[1]
            encoded_sequences.append([encoded_sequence])
            sequence_genes.append([token_gene_map])

            for token_id, gene in zip(encoded_sequence, token_gene_map):
                if token_id == 0:
                    continue
                if gene not in gene_mapping:
                    gene_mapping[gene] = []
                if token_id not in gene_mapping[gene]:
                    gene_mapping[gene].append(token_id)

    # Save mappings only during training
    if split == "train":
        mappings = {'gene_to_token': gene_mapping}
        mapping_filename = f"Gene_Token_Mapping_{tokenizer_type}.json"
        with open(f"{args.save_path}/{args.antibiotic}/{mapping_filename}", "w") as j:
            json.dump(mappings, j, indent=3)
        print(f"Saved {tokenizer_type.upper()} gene-token mappings")

    print(f"{split.capitalize()} tokenization stats - UNK: {unk}, Non-UNK: {non_unk}")

    return list(zip(encoded_sequences, prepped_labels, seq_ids, sequence_genes)), (unk, non_unk)


def create_folds(train_val_data, labels, target_format, n_splits=5, val_size=0.2, seed=42):
    folds = []

    if n_splits == 1:
        if target_format == "multi-cat":
            train_data, val_data = train_test_split(train_val_data, test_size=val_size, random_state=seed)
        else:
            train_data, val_data = train_test_split(train_val_data, test_size=val_size,
                                                    stratify=labels, random_state=seed)
        folds.append((train_data, val_data))
    else:
        if target_format == "multi-cat":
            splitter = ShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=seed)
            for train_idx, val_idx in splitter.split(train_val_data):
                folds.append(([train_val_data[i] for i in train_idx],
                              [train_val_data[i] for i in val_idx]))
        else:
            splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=seed)
            for train_idx, val_idx in splitter.split(train_val_data, labels):
                folds.append(([train_val_data[i] for i in train_idx],
                              [train_val_data[i] for i in val_idx]))

    return folds