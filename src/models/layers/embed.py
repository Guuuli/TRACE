import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.masking import Masking


class RelPosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(RelPosEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()  # [max_len, d_model]
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, seq_len):
        return self.pe[:, : seq_len] #[1, seq_len, d_model]



class Patching(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        if self.stride != self.patch_len:
            warnings.warn(
                "Stride and patch length are not equal. \
                          This may lead to unexpected behavior."
            )

    def forward(self, x):
        '''
        Input:
            x : [B, C, L]
        Output:
            x : [B, C, num_patch, patch_len]
        '''
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return x
  
        
class TimeEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_channels: int = 7,
        patch_len: int = 8,
        stride: int = 8,
        dropout: int = 0.1,
        pos_embed_type: str = "rel_pos",  # "rope" or "rel_pos"
        value_embedding_bias: bool = False,
        orth_gain: float = 1.41,
        channel_special_tokens: bool = True,
    ):
        super(TimeEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_channels = num_channels
        self.pos_embed_type = pos_embed_type
        self.channel_special_tokens = channel_special_tokens
        if self.patch_len > 1:
            self.patching = Patching(patch_len, stride)
            self.value_embedding = nn.Linear(patch_len, d_model, bias=value_embedding_bias)
        else:
            self.value_embedding = nn.Linear(1, d_model, bias=value_embedding_bias)

        self.mask_embedding = nn.Parameter(torch.zeros(d_model))
        # Create learnable special tokens for each channel
        self.channel_tokens = nn.Parameter(torch.zeros(num_channels, d_model))
        nn.init.normal_(self.channel_tokens, std=0.1)  # Initialize with small random values
        self.cls_token = nn.Parameter(torch.zeros(1, d_model))
        
        
        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.value_embedding.weight, gain=orth_gain)
            if value_embedding_bias:
                self.value_embedding.bias.data.zero_()

        # Positional embedding
        if self.pos_embed_type == "rel_pos":
            self.position_embedding = RelPosEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        if self.patch_len > 0:  do patching, otherwise do not patch
        Input:
            x : [B, C, L]
            mask : [B, C, L]
        Output:
            if TraceEncoder:
                x : [B, total_seq_L, d_model] with total_seq_L = C * L + C + 1 or C * num_patch + C + 1
            else:
                x : [B, C, N, d_model]
        """
        # Patching if patch_len > 0
        if self.patch_len > 1:
            x = self.patching(x) # [B, C, num_patch, patch_len]
            mask = Masking.convert_seq_to_patch_view(mask, patch_len=self.patch_len) # [B, C, num_patch]
        else:
            x = x.unsqueeze(-1) # [B, C, L, 1]
        
        mask = mask.unsqueeze(-1).repeat_interleave(self.d_model, dim=-1)
        # mask : [B, C, length, d_model]

        # Input encoding
        x = mask * self.value_embedding(x) + (1 - mask) * self.mask_embedding  # [B, C, length, d_model]
        
        # Positional embedding
        if self.pos_embed_type == "rel_pos":
            x = x + self.position_embedding(seq_len=x.shape[2]) 
            # [B, C, length, d_model]
        if self.channel_special_tokens:
        # Concatenate channels into a single sequence
            seq_length = self.num_channels * x.shape[2] + self.num_channels + 1
            token_sequence = torch.zeros(x.shape[0], seq_length, self.d_model, device=x.device)
            token_sequence[:, 0, :] = self.cls_token
            for v in range(self.num_channels):
                start_idx = 2 + v * (x.shape[2] + 1)
                token_sequence[:, start_idx:start_idx + x.shape[2]] = x[:, v]
                token_sequence[:, start_idx-1, :] = self.channel_tokens[v]
            
            token_sequence = self.dropout(token_sequence)
        else:
            token_sequence = self.dropout(x)
        
        return token_sequence


