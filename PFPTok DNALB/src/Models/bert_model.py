import torch.nn.functional as F
import math
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, debug=False):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.debug = debug

    def forward(self, x, mask=None):
        attn_scores = self.attention(x).squeeze(-1)
        if mask is not None:
            min_value = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(mask == 0, min_value)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.unsqueeze(-1)
        if mask is not None:
            attn_weights = attn_weights * mask.unsqueeze(-1).float()

        output = torch.sum(attn_weights * x, dim=1)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, debug=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.debug = debug

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model, debug=False):
        super(LearnedPositionalEmbedding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.debug = debug

    def forward(self, x):
        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        pos_embed = self.pe(positions)
        output = x + pos_embed
        return output

class MultiHeadAttentionWithRoPEFlash(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_rope=True, use_flash_attention=True, debug=False):
        super(MultiHeadAttentionWithRoPEFlash, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_rope = use_rope
        self.use_flash_attention = use_flash_attention
        self.debug = debug

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
        x_out = torch.cat([x_rotated_first, x_rotated_second], dim=-1)
        return x_out
    
    def forward(self, x, attn_mask=None, key_padding_mask=None, output_attentions=False):
        B, T, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
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
            # Aggregate attention weights over heads for return if needed
            attn_weights = attn_weights.mean(dim=1)  # (B, T, T)
    
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)
    
        return attn_output, attn_weights

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_size, dropout=0.1, use_rope=True, use_flash_attention=True, debug=False):
        super(TransformerBlock, self).__init__()
        self.debug = debug
        self.attention = MultiHeadAttentionWithRoPEFlash(
            d_model, num_heads, dropout, use_rope=use_rope, use_flash_attention=use_flash_attention, debug=debug
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
            x_norm1, 
            attn_mask=None, 
            key_padding_mask=key_padding_mask,
            output_attentions=output_attentions
        )
        x = x + self.dropout(attn_out)
    
        x_norm2 = self.layer_norm2(x)
        ff_out = self.feed_forward(x_norm2)
        x = x + self.dropout(ff_out)
        return x, attn_weights


class BERT(nn.Module):
    def __init__(self, vocab_size, config, paired, debug=False):
        super(BERT, self).__init__()
        self.debug = debug

        hidden_dim      = config['hidden_dim']
        num_heads       = config['num_heads']
        ff_dim          = config['ff_dim']
        num_layers      = config['num_layers']
        self.num_labels = config.get('num_labels', 1)
        dropout         = config.get('dropout', 0.1)
        self.paired     = paired # eQTL mode

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = LearnedPositionalEmbedding(max_len=5000, d_model=hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                ff_size=ff_dim,
                dropout=dropout,
                use_rope=True,
                use_flash_attention=True,
            ) for _ in range(num_layers)
        ])

        self.attention_pooling = AttentionPooling(hidden_dim)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)

        classifier_input_dim = hidden_dim * 6 if self.paired else hidden_dim * 3

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(p=0.3),
            nn.Linear(classifier_input_dim, hidden_dim),
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

    def encode(self, input_ids, attention_mask, output_attentions=False):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.embedding_dropout(x)
        x = self.embedding_norm(x)

        all_attn_weights = [] if output_attentions else None
        for layer in self.layers:
            x, attn_weights = layer(x, mask=attention_mask,
                                    output_attentions=output_attentions)
            if output_attentions and attn_weights is not None:
                all_attn_weights.append(attn_weights)

        attn_pooled  = self.attention_pooling(x, mask=attention_mask)
        x_masked     = x * attention_mask.unsqueeze(-1).float() if attention_mask is not None else x
        x_t          = x_masked.transpose(1, 2)
        max_pooled   = self.max_pooling(x_t).squeeze(-1)
        avg_pooled   = self.avg_pooling(x_t).squeeze(-1)

        pooled = torch.cat([attn_pooled, max_pooled, avg_pooled], dim=-1)
        return pooled, all_attn_weights

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # paired inputs (eQTL mode)
        input_ids_alt=None,
        attention_mask_alt=None,
        labels=None,
        output_attentions=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_hidden_states=None,
    ):
        pooled_ref, attn_weights = self.encode(
            input_ids, attention_mask, output_attentions=output_attentions or False
        )

        if self.paired and input_ids_alt is not None:
            pooled_alt, _ = self.encode(
                input_ids_alt, attention_mask_alt,
                output_attentions=False
            )
            pooled_output = torch.cat([pooled_ref, pooled_alt], dim=-1)
        else:
            pooled_output = pooled_ref

        logits = self.classifier(pooled_output)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=tuple(attn_weights) if output_attentions and attn_weights else None,
        )

