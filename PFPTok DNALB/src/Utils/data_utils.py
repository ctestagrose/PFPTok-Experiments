from .Dataset import SeqDataset
import json
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, train_test_split
from tqdm import tqdm
from pathlib import Path
from .Dataset import SeqDatasetEQTL


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


def create_dataset_eqtl(zipped_data, classification_type):
    ref_seqs  = []
    alt_seqs  = []
    labs      = []
    seq_ids   = []
    gene_ids  = []

    for ref, alt, label, seq_id, gene_map in zipped_data:
        ref_seqs.append(ref[0])   # unwrap outer list
        alt_seqs.append(alt[0])
        labs.append(label)
        seq_ids.append(seq_id)
        gene_ids.append(gene_map)

    return SeqDatasetEQTL(ref_seqs, alt_seqs, labs, seq_ids, gene_ids, classification_type)

def load_json_splits(json_path):
    jp = Path(json_path)
    if not jp.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(jp, "r") as f:
        blob = json.load(f)

    def to_split(name):
        out = []
        split = blob.get(name, {})
        for k, rec in split.items():
            seq = rec["seq"]
            lab = int(rec["binary"])
            seq_id = f"{name}_{k}"
            out.append(([seq], lab, seq_id, []))
        return out

    return to_split("train"), to_split("validation"), to_split("test")


def tokenize_sequences_no_genes_eqtl(prepped_zipped, tokenizer_manager, tokenizer, args):
    """
    eQTL variant: tokenizes (seq_ref, seq_alt) pairs separately.
    Returns encoded as ([ref_ids], [alt_ids], label, seq_id, [gene_map]).
    """
    encoded = []
    unk = 0
    non_unk = 0

    unk_id = tokenizer.unk_token_id
    for seq_ref, seq_alt, label, seq_id in tqdm(prepped_zipped, desc="Tokenizing"):
        ref_ids = tokenizer_manager.encode_sequences(seq_ref, tokenizer)
        alt_ids = tokenizer_manager.encode_sequences(seq_alt, tokenizer)
        if unk_id is not None:
            for t in ref_ids:
                if t == unk_id:
                    unk += 1
                else:
                    non_unk += 1
        gene_map = [-100] * len(ref_ids)  
        encoded.append(([ref_ids], [alt_ids], label, seq_id, [gene_map]))
    return encoded, (unk, non_unk) 


def tokenize_sequences_no_genes(prepped_zipped, tokenizer_manager, tokenizer, args):
    encoded = []
    unk = 0
    non_unk = 0

    unk_id = tokenizer.unk_token_id

    for seqs, label, seq_id in tqdm(prepped_zipped):
        raw_seq = seqs
        ids = tokenizer_manager.encode_sequences(
            raw_seq, 
            tokenizer
        )
        if unk_id is not None:
            for t in ids:
                if t == unk_id:
                    unk += 1
                else:
                    non_unk += 1
        token_gene_map = [-100] * len(ids)
        encoded.append(([ids], label, seq_id, [token_gene_map]))

    return encoded, (unk, non_unk)

def tokenize_sets(tokenizer_manager, prepped_seqs, prepped_labels, seq_ids, genes_list, tokenizer, split, args):
    encoded_sequences = []
    sequence_genes = []
    gene_mapping = {}
    token_to_gene = {} 

    unk = 0
    non_unk = 0

    for index, sequence in enumerate(tqdm(prepped_seqs)):
        genes_in_this_isolate = genes_list[index]
        encoded_sequence, token_gene_map, unk_counts = tokenizer_manager.encode_sequences_genes(sequence, tokenizer, genes_in_this_isolate, split)
        unk+=unk_counts[0]
        non_unk+=unk_counts[1]
        encoded_sequences.append([encoded_sequence])
        sequence_genes.append([token_gene_map])

        for token, gene in zip(encoded_sequence, token_gene_map):
            token_id = token
            if token_id == 0: 
                continue

            if gene not in gene_mapping:
                gene_mapping[gene] = []
            if token_id not in gene_mapping[gene]:
                gene_mapping[gene].append(token_id)
    
    mappings = {
        'gene_to_token': gene_mapping
    }

    
    with open(f"{args.save_path}/{args.antibiotic}/Gene_Token_Mapping.json", "w") as j:
        json.dump(mappings, j, indent=3)


    return list(zip(encoded_sequences, prepped_labels, seq_ids, sequence_genes)), (unk, non_unk)

def create_folds(train_val_data, labels, target_format, n_splits=5, val_size=0.2, seed=42):
    folds = []

    if n_splits == 1:
        if target_format == "multi-cat":
            train_data, val_data = train_test_split(train_val_data,
                                                    test_size=val_size,
                                                    random_state=seed)
        else:
            train_data, val_data = train_test_split(train_val_data,
                                                    test_size=val_size,
                                                    stratify=labels,
                                                    random_state=seed)
        folds.append((train_data, val_data))
    else:
        if target_format == "multi-cat":
            splitter = ShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=seed)
            for train_idx, val_idx in splitter.split(train_val_data):
                train_fold = [train_val_data[i] for i in train_idx]
                val_fold = [train_val_data[i] for i in val_idx]
                folds.append((train_fold, val_fold))
        else:
            splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=seed)
            for train_idx, val_idx in splitter.split(train_val_data, labels):
                train_fold = [train_val_data[i] for i in train_idx]
                val_fold = [train_val_data[i] for i in val_idx]
                folds.append((train_fold, val_fold))

    return folds