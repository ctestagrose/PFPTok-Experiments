import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        attn_scores = self.attention(x).squeeze(-1)  # (batch_size, seq_len)
        if mask is not None:
            min_value = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(mask == 0, min_value)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        if mask is not None:
            attn_weights = attn_weights * mask.unsqueeze(-1).float()
        output = torch.sum(attn_weights * x, dim=1)  # (batch_size, d_model)
        return output


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(LearnedPositionalEmbedding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        return x + self.pe(positions)


class MultiHeadAttentionWithRoPEFlash(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_rope=True, use_flash_attention=True):
        super(MultiHeadAttentionWithRoPEFlash, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_rope = use_rope
        self.use_flash_attention = use_flash_attention

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def apply_rope(self, x):
        if not self.use_rope:
            return x
        B, H, T, D = x.shape
        half_dim = D // 2
        positions = torch.arange(T, device=x.device, dtype=torch.float)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=x.device, dtype=torch.float) / half_dim))
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
        sin = torch.sin(sinusoid_inp).unsqueeze(0).unsqueeze(0)
        cos = torch.cos(sinusoid_inp).unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x_rotated_first = x1 * cos - x2 * sin
        x_rotated_second = x1 * sin + x2 * cos
        return torch.cat([x_rotated_first, x_rotated_second], dim=-1)

    def forward(self, x, attn_mask=None, key_padding_mask=None, output_attentions=False):
        B, T, _ = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.apply_rope(q)
        k = self.apply_rope(k)

        use_flash = self.use_flash_attention and not output_attentions

        if use_flash and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            if key_padding_mask is not None:
                additive_mask = torch.zeros((B, T), dtype=q.dtype, device=q.device)
                additive_mask = additive_mask.masked_fill(key_padding_mask, float('-inf'))
                additive_mask = additive_mask.view(B, 1, 1, T)
            else:
                additive_mask = None

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=additive_mask,
                dropout_p=0.0,
                is_causal=False
            )
            attn_weights = None
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
            attn_weights = attn_weights.mean(dim=1)  # Average over heads: (B, T, T)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_size, dropout=0.1, use_rope=True, use_flash_attention=True):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttentionWithRoPEFlash(
            d_model, num_heads, dropout, use_rope=use_rope, use_flash_attention=use_flash_attention
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_size),
            nn.GELU(),
            nn.Linear(ff_size, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None, output_attentions=False):
        key_padding_mask = (mask == 0) if mask is not None else None

        x_norm1 = self.layer_norm1(x)
        attn_out, attn_weights = self.attention(
            x_norm1, attn_mask=None, key_padding_mask=key_padding_mask,
            output_attentions=output_attentions
        )
        x = x + self.dropout(attn_out)

        x_norm2 = self.layer_norm2(x)
        ff_out = self.feed_forward(x_norm2)
        x = x + self.dropout(ff_out)
        return x, attn_weights


class BERT(nn.Module):
    def __init__(self, vocab_size, config):
        super(BERT, self).__init__()

        hidden_dim = config['hidden_dim']
        num_heads = config['num_heads']
        ff_dim = config['ff_dim']
        num_layers = config['num_layers']
        self.num_labels = config.get('num_labels', 1)
        dropout = config.get('dropout', 0.1)
        self.grad_clip = 1.0

        # Token + Positional Embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = LearnedPositionalEmbedding(max_len=5000, d_model=hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(hidden_dim)

        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=hidden_dim, num_heads=num_heads, ff_size=ff_dim,
                dropout=dropout, use_rope=True, use_flash_attention=True
            ) for _ in range(num_layers)
        ])

        # Pooling Strategies
        self.attention_pooling = AttentionPooling(hidden_dim)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim // 2, self.num_labels)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_hidden_states=None
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            x = self.embedding(input_ids)
        elif inputs_embeds is not None:
            x = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        internal_mask = attention_mask

        x = self.positional_encoding(x)
        x = self.embedding_dropout(x)
        x = self.embedding_norm(x)

        # Transformer Layers
        all_attn_weights = [] if output_attentions else None
        for layer in self.layers:
            layer_output = layer(x, mask=internal_mask, output_attentions=output_attentions)
            x = layer_output[0]
            if output_attentions and layer_output[1] is not None:
                all_attn_weights.append(layer_output[1])

        # Pooling
        attention_output = self.attention_pooling(x, mask=internal_mask)
        x_masked = x * internal_mask.unsqueeze(-1).float() if internal_mask is not None else x
        x_transpose = x_masked.transpose(1, 2)
        max_pooled = self.max_pooling(x_transpose).squeeze(-1)
        avg_pooled = self.avg_pooling(x_transpose).squeeze(-1)

        pooled_output = torch.cat([attention_output, max_pooled, avg_pooled], dim=-1)

        # Classification (loss computed by CustomTrainer)
        logits = self.classifier(pooled_output)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=tuple(all_attn_weights) if output_attentions and all_attn_weights else None,
        )