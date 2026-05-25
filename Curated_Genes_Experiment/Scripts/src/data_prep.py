
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

from tqdm import tqdm

def collect_ordered_fastas(root: str, suffix: str = "H37Rv_ordered") -> list[str]:
    exts = (".fasta", ".fa", ".fna")
    files = []
    for p in Path(root).rglob(f"*.{suffix}*"):
        if p.suffix.lower() in exts:
            files.append(str(p))
    return sorted(files)

def _strip_ordered_suffix(stem: str, suffix: str = "H37Rv_ordered") -> str:
    if stem.endswith("." + suffix):
        return stem[:-(len(suffix) + 1)]
    return stem


def _read_scaffolds(fpath: str) -> Tuple[List[str], List[str]]:
    seqs: List[str] = []
    headers: List[str] = []
    with open(fpath, "r") as fh:
        cur: List[str] = []
        head: str = None
        for line in fh:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur).upper())
                    headers.append(head if head is not None else f"SCAF_{len(seqs)}")
                    cur = []
                head = line[1:].strip().split()[0]
            else:
                cur.append(line.strip())
        if cur:
            seqs.append("".join(cur).upper())
            headers.append(head if head is not None else f"SCAF_{len(seqs)}")
    return seqs, headers


def _concat_with_spacer(seqs: List[str], spacer_ns: int) -> str:
    if not seqs:
        return ""
    if spacer_ns and spacer_ns > 0:
        spacer = "N" * spacer_ns
        return spacer.join(seqs)
    return "".join(seqs)


def _resolve_fasta_path(fpath: str, args) -> str:
    p = Path(fpath)
    if p.is_dir():
        ordered_suffix = getattr(args, 'ordered_suffix', 'H37Rv_ordered')
        exts = (".fasta", ".fa", ".fna")
        candidates = [q for q in p.iterdir()
                      if q.is_file()
                      and q.suffix.lower() in exts
                      and q.stem.endswith("." + ordered_suffix)]
        if len(candidates) == 1:
            return str(candidates[0])
        cand = p / getattr(args, 'scaffold_filename', 'scaffolds.fasta')
        if cand.exists():
            return str(cand)
    return str(p)



def _infer_isolate_id(fasta_path: str, args) -> str:
    p = Path(fasta_path)
    isolate = p.stem
    ordered_suffix = getattr(args, 'ordered_suffix', 'H37Rv_ordered')
    isolate = _strip_ordered_suffix(isolate, ordered_suffix)
    if getattr(args, 'use_scaffolds', False):
        if getattr(args, 'isolate_id_from_dir', False) or p.name == getattr(args, 'scaffold_filename', 'scaffolds.fasta'):
            isolate = p.parent.name

    if "_" in isolate:
        isolate = isolate.split("_")[0]
    return isolate



def _build_label(isolate: str, targets: Dict[str, Any], target_format: str) -> Any:
    if isolate not in targets:
        return None
    else:
        ld = targets[isolate]
        return ld.get(target_format, None)


def _label_is_valid(label: Any) -> bool:
    if label is None:
        return False
    try:
        return not math.isnan(float(label))
    except Exception:
        return False


def _read_sequences_for_file(fasta_path: str, args, target_format: str, gene_manager) -> Tuple[List[str], List[str]]:
    if getattr(args, 'use_scaffolds', False):
        scaf_seqs, scaf_headers = _read_scaffolds(fasta_path)
        spacer_ns = getattr(args, 'scaffold_spacer_ns', 200)
        concat_seq = _concat_with_spacer(scaf_seqs, spacer_ns)
        return [concat_seq], ["SCAF_ALL"]

    seqdir = getattr(args, 'sequence_dir', '')
    if ("Merged" in seqdir) or ("10_23" in seqdir):
        if getattr(args, 'use_gene_file', False):
            return gene_manager.read_fasta_file_genes(fasta_path, target_format)
        else:
            return gene_manager.read_fasta_file_genes_all(fasta_path, target_format)
    else:
        return gene_manager.read_fasta_file_genes(fasta_path, target_format)


class DataPreparer:
    def __init__(self, gene_manager, sequence_processor, tokenizer_manager, args):
        self.gene_manager = gene_manager
        self.sequence_processor = sequence_processor
        self.tokenizer_manager = tokenizer_manager
        self.args = args

    def prep_data(self, fasta_files, full_set, target_file, target_format, mode):
        sequences: List[List[str]] = []
        labels: List[Any] = []
        seq_ids: List[str] = []
        genes_list: List[List[str]] = []

        with open(target_file, "r") as f:
            targets = json.load(f)

        if mode in ("Evaluate", "Train"):
            split_dict = {}
            for item, val in targets.items():
                if "." in item:
                    for err in item.split("."):
                        split_dict[err] = val
                else:
                    split_dict[item] = val
            targets = split_dict

        for original_path in tqdm(fasta_files, desc="Processing FASTA", unit="file"):
            fasta_path = _resolve_fasta_path(original_path, self.args)
            isolate = _infer_isolate_id(fasta_path, self.args)
            if isolate not in targets:
                continue

            label = _build_label(isolate, targets, target_format)
            if not _label_is_valid(label):
                continue

            clean_sequence, genes_in_isolate = _read_sequences_for_file(fasta_path, self.args, target_format, self.gene_manager)
            if not clean_sequence or not clean_sequence[0]:
                continue

            sequences.append(clean_sequence)
            labels.append(label)
            seq_ids.append(isolate)
            genes_list.append(genes_in_isolate)

        # === full_set for tokenizer ===
        full_set_seqs: List[List[str]] = []
        for original_path in full_set:
            fasta_path = _resolve_fasta_path(original_path, self.args)
            isolate = _infer_isolate_id(fasta_path, self.args)
            if isolate not in targets:
                continue
            clean_sequence, _ = _read_sequences_for_file(fasta_path, self.args, target_format, self.gene_manager)
            if clean_sequence and clean_sequence[0]:
                full_set_seqs.append(clean_sequence)

        zipped_data = list(zip(sequences, labels, seq_ids, genes_list))
        return zipped_data, full_set_seqs

