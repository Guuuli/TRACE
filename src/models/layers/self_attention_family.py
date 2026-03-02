from math import sqrt

import torch
import torch.nn as nn
from einops import repeat
from src.models.layers.attn_bias import BinaryAttentionBias
from src.utils.masking import Masking, TRACEMask
from src.models.layers.attn_projection import QueryKeyProjection, RotaryProjection



class TraceAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False, d_model=512, num_heads=8, mask_flag=True, flash_attention=False, pos_embed_type="rope"):
        super(TraceAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.flash_attention = flash_attention
        self.pos_embed_type = pos_embed_type
        self.qk_proj = QueryKeyProjection(dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection,partial_factor=(0.0, 0.5))
        self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=num_heads)

    def forward(self, queries, keys, values, attn_mask, **kwargs):
        n_vars = kwargs["n_vars"]
        n_tokens = kwargs["n_tokens"]
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
    
        # [B, H, L, E]
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        full_seq = torch.cat([torch.tensor([0]), torch.arange(0, n_tokens+1).repeat(n_vars)]).to(queries.device)
        seq_id = repeat(full_seq, 'n -> b h n', b=B, h=H)   #[B, H, total_len]
        assert seq_id.shape == (B, H, L)
        if self.pos_embed_type == "rope":
            queries, keys = self.qk_proj(
                queries, keys, query_id=seq_id, kv_id=seq_id)

        scale = self.scale or 1. / sqrt(E)


        if self.mask_flag:
            if attn_mask is None:
                attn_mask_model = TRACEMask(B, n_vars, n_tokens, device=queries.device) #[B, 1, total_len, total_len]
                attn_mask = attn_mask_model.mask
            else:
                attn_mask_model = TRACEMask(B, n_vars, n_tokens, device=queries.device)
                attn_mask = Masking.mask_patch_to_seq_with_special_tokens(attn_mask)  #[B, total_len]
                attn_mask = Masking.mask_seq_to_attention(attn_mask, attn_mask_model.mask)  #[B, 1, total_len, total_len]
                
            attn_mask = attn_mask.masked_fill(attn_mask, float("-inf"))


        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            scores += attn_mask
            
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None

class ResidualCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=False)
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout)
        if use_layernorm:
            self.ln = nn.LayerNorm(d_model)

    def forward(self, x_query, x_kv):
        """
        Args:
            x_query: [B, C, D]   # e.g., time series embedding
            x_kv:    [B, C, D]   # e.g., text embedding
        Returns:
            out: [B, C, D]
        """
        # [B, C, D] -> [C, B, D]
        q = x_query.transpose(0, 1)
        k = x_kv.transpose(0, 1)
        v = x_kv.transpose(0, 1)

        # Cross-Attention
        attn_output, _ = self.cross_attn(q, k, v)  # [C, B, D]
        attn_output = self.dropout(attn_output)

        # Residual + optional LayerNorm
        out = q + attn_output  # [C, B, D]
        out = out.transpose(0, 1)  # [B, C, D]
        if self.use_layernorm:
            out = self.ln(out)
        return out # [B, C, D]
    
    
class AttentionLayer(nn.Module):
    """Should be compatible with both AnomalyTransformer and vanilla transformer"""
    def __init__(
        self,
        attention,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None
    ):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.norm = nn.LayerNorm(d_model)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, **kwargs):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, **kwargs
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
    
