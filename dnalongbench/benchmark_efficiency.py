import argparse
import glob
import gzip
import json
import os
import time

import torch
import torch.nn as nn


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def load_split(json_path, split):
    root = json_path.rstrip("/")
    stem = os.path.basename(root)
    cands = []
    for base in (f"{root}_{split}", os.path.join(root, split),
                 os.path.join(root, f"{stem}_{split}")):
        cands += [base + e for e in (".jsonl.gz", ".jsonl")]
    for d in (root, os.path.dirname(root)):
        cands += sorted(glob.glob(os.path.join(d, f"*{split}.jsonl.gz")))
    for p in cands:
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt") as f:
                return p, [json.loads(l) for l in f if l.strip()]
    raise FileNotFoundError(f"no '{split}' split for {json_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["bert", "hyena", "caduceus", "ntv3"],
                    help="Model architecture to benchmark.")
    ap.add_argument("--json_path", required=True,
                    help="Path to the DNALongBench JSON split directory.")
    ap.add_argument("--model_config", default=None,
                    help="Path to BERT model config JSON (required for --model bert).")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="Stop after this many gradient steps (0 = full epochs).")
    ap.add_argument("--seq_len", type=int, default=450_000,
                    help="Maximum sequence length.")
    ap.add_argument("--pfp_w", type=int, default=100,
                    help="PFPTok window size w (BERT only).")
    ap.add_argument("--pfp_p", type=int, default=4096,
                    help="PFPTok hash modulus p (BERT only).")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--bf16", action="store_true",
                    help="Enable bfloat16 autocast during training.")
    ap.add_argument("--out", default="efficiency_results.json",
                    help="Output file — results are appended as a JSON list.")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "Run on the GPU you report in the paper."
    dev = torch.device("cuda")

    src, records = load_split(args.json_path, "train")
    key = next(k for k in ("seq", "sequence", "seq_ref") if k in records[0])
    seqs = [r[key] for r in records]
    labels = [int(r["label"]) for r in records]
    print(f"{len(seqs)} records from {os.path.basename(src)}, "
          f"mean {sum(map(len, seqs))//len(seqs)} bp")

    triples = list(zip(seqs, labels, range(len(seqs))))
    pad_id, tok_train_s = 0, 0.0

    t0 = time.time()

    if args.model == "bert":
        assert args.model_config, "--model_config is required for --model bert"
        from utils.pfptok import PFPTok
        from transformers import PreTrainedTokenizerFast
        from models.bert_model import BERT

        pfp = PFPTok()
        print(f"Training PFP tokenizer (w={args.pfp_w}, p={args.pfp_p})...")
        t_tok = time.time()
        raw = pfp.setup_tokenizer(seqs, w=args.pfp_w, p=args.pfp_p)
        tok_train_s = time.time() - t_tok
        wrapped = PreTrainedTokenizerFast(
            tokenizer_object=raw, bos_token="[CLS]", eos_token="[SEP]",
            unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]",
            cls_token="[CLS]", mask_token="[MASK]",
            additional_special_tokens=["[INTB]", "[INTA]", "[GENE]"],
        )
        pad_id = wrapped.pad_token_id
        vocab_size = wrapped.vocab_size
        print(f"PFP tokenizer trained in {tok_train_s:.1f}s, vocab {vocab_size}")

        ids = [pfp.encode_sequences(s, wrapped) for s, _, _ in triples]

        cfg = json.load(open(args.model_config))
        cfg["num_labels"] = 1
        model = BERT(vocab_size, cfg, paired=False)

    elif args.model == "hyena":
        from utils.hyena_tokenizer import build_hyena_tokenizer, tokenize_sequences_hyena
        from models.hyena_model import HyenaDNAForClassification

        tok = build_hyena_tokenizer(model_max_length=args.seq_len)
        enc, _ = tokenize_sequences_hyena(triples, tok, max_length=args.seq_len)
        ids = [e[0][0] for e in enc]
        pad_id = tok.pad_token_id
        model = HyenaDNAForClassification(num_labels=1, paired=False)

    elif args.model == "caduceus":
        from utils.caduceus_tokenizer import build_caduceus_tokenizer, tokenize_sequences_caduceus
        from models.caduceus_model import CaduceusForClassification

        tok = build_caduceus_tokenizer(model_max_length=args.seq_len)
        enc, _ = tokenize_sequences_caduceus(triples, tok, max_length=args.seq_len)
        ids = [e[0][0] for e in enc]
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 4
        model = CaduceusForClassification(num_labels=1, paired=False)

    else:
        from utils.ntv3_tokenizer import build_ntv3_tokenizer, tokenize_sequences_ntv3
        from models.ntv3_model import NTv3ForClassification

        tok = build_ntv3_tokenizer()
        enc, _ = tokenize_sequences_ntv3(triples, tok, max_length=args.seq_len, length_mode="pad")
        ids = [e[0][0] for e in enc]
        pad_id = tok.get_vocab().get("N", tok.get_vocab().get("n", 0))
        model = NTv3ForClassification(num_labels=1, paired=False)

    tok_total_s = time.time() - t0
    lens = [len(x) for x in ids]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"tokenized in {tok_total_s:.1f}s | tokens/seq mean {sum(lens)/len(lens):.0f} "
          f"(min {min(lens)}, max {max(lens)}) | {n_params/1e6:.1f}M params")

    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ys = torch.tensor(labels, dtype=torch.float)
    lossf = nn.BCEWithLogitsLoss()

    def make_batch(lo, hi):
        chunk = ids[lo:hi]
        m = max(len(c) for c in chunk)
        out = torch.full((len(chunk), m), pad_id, dtype=torch.long)
        msk = torch.zeros((len(chunk), m), dtype=torch.long)
        for r, c in enumerate(chunk):
            out[r, : len(c)] = torch.tensor(c, dtype=torch.long)
            msk[r, : len(c)] = 1
        return out.to(dev, non_blocking=True), msk.to(dev, non_blocking=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    epoch_times, steps = [], 0
    model.train()

    for ep in range(args.epochs):
        t_ep = time.time()
        for i in range(0, len(ids), args.batch_size):
            x, mask = make_batch(i, i + args.batch_size)
            y = ys[i : i + args.batch_size].to(dev)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                out = model(input_ids=x, attention_mask=mask)
            loss = lossf(out.logits.float().squeeze(-1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break

        torch.cuda.synchronize()
        dt = time.time() - t_ep
        epoch_times.append(dt)
        print(f"  epoch {ep+1}: {dt:.1f}s")
        if args.max_steps and steps >= args.max_steps:
            break

    peak = torch.cuda.max_memory_allocated()
    steady = epoch_times[1:] or epoch_times
    mean_ep = sum(steady) / len(steady)
    seen = (min(len(ids), args.max_steps * args.batch_size)
            if args.max_steps else len(ids))

    res = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "params_M": round(n_params / 1e6, 2),
        "input_bp": args.seq_len,
        "tokens_per_seq_mean": round(sum(lens) / len(lens), 1),
        "tokens_per_seq_max": max(lens),
        "batch_size": args.batch_size,
        "bf16": args.bf16,
        "tokenizer_train_seconds": round(tok_train_s, 1),
        "tokenize_seconds_total": round(tok_total_s, 1),
        "epoch_seconds_all": [round(t, 1) for t in epoch_times],
        "epoch_seconds_mean_excl_first": round(mean_ep, 1),
        "samples_per_second": round(seen / mean_ep, 3),
        "peak_gpu_alloc_bytes": peak,
        "peak_gpu_alloc_human": human(peak),
    }
    print("\n" + json.dumps(res, indent=2))

    prior = json.load(open(args.out)) if os.path.exists(args.out) else []
    prior.append(res)
    json.dump(prior, open(args.out, "w"), indent=2)
    print(f"\nappended to {args.out}")


if __name__ == "__main__":
    main()
