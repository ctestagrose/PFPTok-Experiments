from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


def _load_caduceus_backbone(model_name: str):
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    try:
        backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, config=config
        )
    except (ValueError, KeyError, OSError):
        mlm = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True, config=config
        )
        backbone = mlm.caduceus
    return config, backbone


class CaduceusForClassification(nn.Module):
    def __init__(
        self,
        model_name: str = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        num_labels: int = 1,
        pooling: str = "mean",
        freeze_backbone: bool = False,
        pos_weight: Optional[float] = None,
        conjoin_eval: bool = False,
        conjoin_train: bool = False,
        paired: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        if pooling not in ("mean", "max", "last", "first"):
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.config, self.caduceus = _load_caduceus_backbone(model_name)
        self.d_model = self.config.d_model
        self.rcps = bool(getattr(self.config, "rcps", False))
        self.pooling = pooling
        self.num_labels = num_labels
        self.paired = bool(paired)

        self.conjoin_eval = bool(conjoin_eval) and not self.rcps
        self.conjoin_train = bool(conjoin_train) and not self.rcps

        in_dim = self.d_model * 2 if self.paired else self.d_model
        self.score = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, num_labels),
        )

        if freeze_backbone:
            for p in self.caduceus.parameters():
                p.requires_grad = False

        if pos_weight is not None:
            self.register_buffer(
                "pos_weight", torch.tensor(float(pos_weight)), persistent=False
            )
        else:
            self.pos_weight = None

        cmap = getattr(self.config, "complement_map", None)
        if cmap is not None:
            size = max(int(k) for k in cmap.keys()) + 1
            lut = torch.arange(size, dtype=torch.long)
            for k, v in cmap.items():
                lut[int(k)] = int(v)
            self.register_buffer("complement_lut", lut, persistent=False)
        else:
            self.complement_lut = None

    def _reverse_complement(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.complement_lut is None:
            raise RuntimeError(
                "config.complement_map is missing; cannot build reverse complements. "
                "Either set conjoin_eval=False or add complement_map to the config."
            )
        return self.complement_lut[input_ids].flip(dims=[1])

    def _pool(self, hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        seq_dim = 1
        if self.pooling == "first":
            return hidden[:, 0]
        if self.pooling == "last":
            if mask is None:
                return hidden[:, -1]
            idx = mask.long().sum(dim=1) - 1
            return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]

        if mask is None:
            if self.pooling == "mean":
                return hidden.mean(dim=seq_dim)
            return hidden.max(dim=seq_dim).values

        m = mask.to(hidden.dtype)
        while m.dim() < hidden.dim():
            m = m.unsqueeze(-1)
        if self.pooling == "mean":
            return (hidden * m).sum(dim=seq_dim) / m.sum(dim=seq_dim).clamp(min=1.0)
        return hidden.masked_fill(m == 0, torch.finfo(hidden.dtype).min).max(dim=seq_dim).values

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.caduceus(input_ids=input_ids, return_dict=True)
        return out.last_hidden_state

    def _hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        conjoin = self.conjoin_train if self.training else (
            self.conjoin_train or self.conjoin_eval
        )
        if self.rcps:
            h = self._encode(input_ids)
            d = self.d_model
            return torch.stack(
                [h[..., :d], torch.flip(h[..., d:], dims=[1, 2])], dim=-1
            )
        if conjoin:
            h_fwd = self._encode(input_ids)
            h_rc = self._encode(self._reverse_complement(input_ids))
            return torch.stack([h_fwd, h_rc.flip(dims=[1])], dim=-1)
        return self._encode(input_ids)

    def _rep(self, input_ids, mask):
        pooled = self._pool(self._hidden(input_ids), mask)
        if pooled.dim() == 3:
            pooled = pooled.mean(dim=-1)
        return pooled

    @staticmethod
    def _find_pair(input_ids, kwargs):
        for a, b in (
            ("ref_ids", "alt_ids"),
            ("input_ids", "alt_ids"),
            ("input_ids", "input_ids_alt"),
            ("input_ids", "input_ids_2"),
            ("ref_input_ids", "alt_input_ids"),
            ("input_ids_ref", "input_ids_alt"),
        ):
            first = input_ids if a == "input_ids" else kwargs.get(a)
            second = kwargs.get(b)
            if first is not None and second is not None:
                return first, second, kwargs.get("attention_mask"), kwargs.get(
                    b.replace("ids", "attention_mask").replace("input_", ""),
                    kwargs.get("attention_mask_alt"),
                )

        pool = ([] if input_ids is None else [input_ids]) + [
            v for k, v in kwargs.items()
            if torch.is_tensor(v)
            and v.dim() == 2
            and v.dtype in (torch.long, torch.int)
            and "mask" not in k
            and "label" not in k
        ]
        if len(pool) >= 2:
            return pool[0], pool[1], kwargs.get("attention_mask"), None
        raise RuntimeError(
            "paired=True but only one sequence tensor was found in the batch. "
            f"Batch keys: {sorted(kwargs)}. Point _find_pair at the right names."
        )

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.paired:
            ref_ids, alt_ids, ref_mask, alt_mask = self._find_pair(input_ids, kwargs)
            if attention_mask is not None and ref_mask is None:
                ref_mask = attention_mask
            rep = torch.cat(
                [self._rep(ref_ids, ref_mask), self._rep(alt_ids, alt_mask)], dim=-1
            )
            logits = self.score(rep)
        else:
            hidden = self._hidden(input_ids)
            pooled = self._pool(hidden, attention_mask)
            if pooled.dim() == 3:  # (B, D, 2) -> shared head, averaged
                logits = (self.score(pooled[..., 0]) + self.score(pooled[..., 1])) / 2
            else:
                logits = self.score(pooled)

        logits = logits.float()

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.view(-1),
                    labels.view(-1).float(),
                    pos_weight=(
                        self.pos_weight.to(logits.device)
                        if self.pos_weight is not None
                        else None
                    ),
                )
            elif labels.dtype in (torch.long, torch.int):
                loss = nn.functional.cross_entropy(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.float()
                )

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def gradient_checkpointing_enable(self, **kwargs):
        pass