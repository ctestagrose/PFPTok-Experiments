import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class HyenaDNAForClassification(nn.Module):
    def __init__(
        self,
        model_name: str = "LongSafari/hyenadna-medium-450k-seqlen-hf",
        num_labels: int = 1,
        pooling: str = "mean",
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        pos_weight: float = None,
        paired: bool = False,       # eQTL mode
    ):
        super().__init__()
        self.num_labels = num_labels
        self.pooling    = pooling
        self.paired     = paired

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_name, config=config, trust_remote_code=True
        )
        hidden_size = config.d_model

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        classifier_input = hidden_size * 2 if paired else hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Dropout(dropout),
            nn.Linear(classifier_input, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

        self.config = config
        self.config.num_labels = num_labels

        if pos_weight is not None:
            self.register_buffer(
                'pos_weight', torch.tensor([pos_weight], dtype=torch.float)
            )
        else:
            self.pos_weight = None

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "last":
            lengths = attention_mask.sum(dim=1) - 1
            lengths = lengths.clamp(min=0)
            idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
            return hidden_states.gather(1, idx).squeeze(1)
        else:
            mask_exp = attention_mask.unsqueeze(-1).float()
            summed   = (hidden_states * mask_exp).sum(dim=1)
            lengths  = mask_exp.sum(dim=1).clamp(min=1e-9)
            return summed / lengths

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids)
        hidden  = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        if attention_mask is None:
            attention_mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=hidden.device)

        return self.pool(hidden, attention_mask)  # (B, H)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        input_ids_alt: torch.Tensor = None,        # eQTL alt sequence
        attention_mask_alt: torch.Tensor = None,   # eQTL alt mask
        labels: torch.Tensor = None,
        **kwargs,
    ):
        pooled_ref = self.encode(input_ids, attention_mask)  # (B, H)

        if self.paired and input_ids_alt is not None:
            pooled_alt = self.encode(input_ids_alt, attention_mask_alt)  # (B, H)
            # concatenate ref and alt — matches HyenaDNA paper approach
            pooled = torch.cat([pooled_ref, pooled_alt], dim=-1)         # (B, H*2)
        else:
            pooled = pooled_ref                                           # (B, H)

        logits = self.classifier(pooled)  # (B, num_labels)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(
                    logits.squeeze(-1), labels.float()
                )
            else:
                loss = nn.CrossEntropyLoss()(logits, labels.long())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )