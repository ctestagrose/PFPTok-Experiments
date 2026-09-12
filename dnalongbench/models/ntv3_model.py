from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


def _load_ntv3_backbone(model_name: str, n_transformer_layers_hint: int = None):
    from transformers import AutoConfig, AutoModelForMaskedLM

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    n_layers = (
        n_transformer_layers_hint
        or getattr(config, "num_layers", None)
        or getattr(config, "num_hidden_layers", None)
        or 6
    )
    try:
        config.embeddings_layers_to_save = (n_layers,)
    except Exception:
        pass

    model = AutoModelForMaskedLM.from_pretrained(
        model_name, config=config, trust_remote_code=True
    )
    return config, model


class NTv3ForClassification(nn.Module):
    def __init__(
        self,
        model_name: str = "InstaDeepAI/NTv3_100M_pre",
        num_labels: int = 1,
        pooling: str = "mean",
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        pos_weight: Optional[float] = None,
        paired: bool = False,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.pooling = pooling
        self.paired = bool(paired)

        self.config, self.ntv3 = _load_ntv3_backbone(model_name)
        self.core = getattr(self.ntv3, "core", self.ntv3)

        self.d_model = (
            getattr(self.config, "embed_dim", None)
            or getattr(self.config, "hidden_size", None)
            or getattr(self.config, "model_dim", None)
            or 768
        )
        self._emb_key = None

        if freeze_backbone:
            for p in self.ntv3.parameters():
                p.requires_grad_(False)

        classifier_input = self.d_model * 2 if self.paired else self.d_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Dropout(dropout),
            nn.Linear(classifier_input, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, num_labels),
        )

        if pos_weight is not None:
            self.register_buffer(
                "pos_weight", torch.tensor([float(pos_weight)]), persistent=False
            )
        else:
            self.pos_weight = None


    def _pick_hidden(self, out) -> torch.Tensor:
        if isinstance(out, dict):
            if self._emb_key is None:
                cands = [
                    k for k, v in out.items()
                    if k.startswith("embeddings_")
                    and not k.startswith("embeddings_deconv")
                    and torch.is_tensor(v) and v.dim() == 3
                ]
                if cands:
                    self._emb_key = sorted(
                        cands, key=lambda k: int(k.rsplit("_", 1)[-1])
                    )[-1]
            if self._emb_key and self._emb_key in out:
                return out[self._emb_key]
            for k in ("last_hidden_state", "hidden_states"):
                v = out.get(k)
                if torch.is_tensor(v) and v.dim() == 3:
                    return v
                if isinstance(v, (list, tuple)) and v:
                    return v[-1]
            raise RuntimeError(
                "Could not find a 3-D hidden tensor in NTv3 output. "
                f"Keys: {sorted(out)}"
            )

        hs = getattr(out, "hidden_states", None)
        if hs:
            return min((h for h in hs if h.dim() == 3), key=lambda h: h.shape[1])
        h = getattr(out, "last_hidden_state", None)
        if h is not None:
            return h
        raise RuntimeError(f"Unexpected NTv3 output type: {type(out)}")

    def _pool(self, hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0, :]

        if mask is not None and mask.shape[1] != hidden.shape[1]:
            if mask.shape[1] % hidden.shape[1] == 0:
                factor = mask.shape[1] // hidden.shape[1]
                mask = mask[:, ::factor][:, : hidden.shape[1]]
            else:
                mask = None

        if self.pooling == "last":
            if mask is None:
                return hidden[:, -1, :]
            idx = mask.long().sum(dim=1).clamp(min=1) - 1
            return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]

        if mask is None:
            return hidden.mean(dim=1) if self.pooling == "mean" else hidden.max(dim=1).values

        m = mask.unsqueeze(-1).to(hidden.dtype)
        if self.pooling == "mean":
            return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        return hidden.masked_fill(m == 0, torch.finfo(hidden.dtype).min).max(dim=1).values

    def encode(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        out = self.core(input_ids=input_ids, output_hidden_states=True)
        return self._pool(self._pick_hidden(out), attention_mask)


    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        input_ids_alt: torch.Tensor = None,
        attention_mask_alt: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        pooled_ref = self.encode(input_ids, attention_mask)

        if self.paired and input_ids_alt is not None:
            pooled_alt = self.encode(input_ids_alt, attention_mask_alt)
            pooled = torch.cat([pooled_ref, pooled_alt], dim=-1)
        else:
            pooled = pooled_ref

        logits = self.classifier(pooled).float()

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(
                    logits.squeeze(-1), labels.float()
                )
            else:
                loss = nn.CrossEntropyLoss()(logits, labels.long())

        return SequenceClassifierOutput(loss=loss, logits=logits)