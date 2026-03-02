import torch
from typing import Optional

class Masking:
    def __init__(
        self, mask_ratio: float = 0.3, patch_len: int = 8, stride: Optional[int] = None
    ):
        """
        Class to generate a mask for Masked Token Reconstruction.
        Indices with 0 mask are hidden, and with 1 are observed.
         mask_ratio=0.3 means 30% is 0 (masked), 70% is 1 (observed)
        """
        self.mask_ratio = mask_ratio
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride

    @staticmethod
    def convert_seq_to_patch_view(
        mask: torch.Tensor, patch_len: int = 8, stride: Optional[int] = None
    ):
        """
        Input:
            mask : torch.Tensor of shape [B, C, seq_len]
        Output
            mask : torch.Tensor of shape [B, C, n_patches]
        """
        stride = patch_len if stride is None else stride
        mask = mask.unfold(dimension=-1, size=patch_len, step=stride)
        # mask : [batch_size, n_channels, n_patches, patch_len]
        return (mask.sum(dim=-1) == patch_len).long()

    @staticmethod
    def convert_patch_to_seq_view(
        mask: torch.Tensor,
        patch_len: int = 8,
    ):
        """
        Input:
            mask : torch.Tensor of shape [B, C, n_patches]
        Output:
            mask : torch.Tensor of shape [B, C, <=seq_len]
        """
        return mask.repeat_interleave(patch_len, dim=-1)

    def generate_mask(self, x: torch.Tensor, input_mask: Optional[torch.Tensor] = None):
        """
        Input:
            x : torch.Tensor of shape [B, C, n_patches, patch_len] or [B, C, seq_len]
            input_mask: torch.Tensor of shape [B, C, seq_len]
        Output:
            mask : torch.Tensor of shape  [B, C, n_patches] or [B, C, seq_len]
        """
        if x.ndim == 4:
            return self._mask_patch_view(x, input_mask=input_mask)
        elif x.ndim == 3:
            return self._mask_seq_view(x, input_mask=input_mask)

    def _mask_patch_view(self, x, input_mask=None):
        """
        Input:
            x : torch.Tensor of shape [B, C, n_patches, patch_len]
            input_mask: torch.Tensor of shape [B, C, seq_len]
        Output:
            mask : torch.Tensor of shape [B, C, n_patches]
        """
        input_mask = self.convert_seq_to_patch_view(
            input_mask, self.patch_len, self.stride
        )  # input_mask: [B, C, n_patches]
        n_observed_patches = input_mask.sum(dim=-1, keepdim=True)  # [B, C, 1]
        B, C, n_patches, _ = x.shape
        len_keep = torch.ceil(n_observed_patches * (1 - self.mask_ratio)).long()  # [B, C, 1]
        noise = torch.rand(
            B, C, n_patches, device=x.device
        )  # noise in [0, 1], B x C x n_patches
        
        position_bias = torch.linspace(0, 1, n_patches, device=x.device)  # [n_patches]
        noise += position_bias.view(1, 1, -1)  # broadcast to [B, C, n_patches]
        
        noise = torch.where(input_mask == 1, noise, torch.ones_like(noise) * 10.0)
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=-1)  # Ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [B, C, n_patches] 
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([B, C, n_patches], device=x.device)  # mask: [batch_size x n_patches]
        for i in range(B):
            for j in range(C):
                mask[i, j, : len_keep[i, j]] = 1

        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=-1, index=ids_restore)

        return mask.long()

    def _mask_seq_view(self, x, input_mask=None):
        """
        Input:
            x : torch.Tensor of shape [B, C, seq_len]
            input_mask: torch.Tensor of shape [B, C, seq_len]
        Output:
            mask : torch.Tensor of shape [B, C, seq_len]
        """
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [B, C, n_patches, patch_len]
        mask = self._mask_patch_view(x, input_mask=input_mask)
        return self.convert_patch_to_seq_view(mask, self.patch_len).long()
    
    
    @staticmethod
    def mask_seq_to_attention(input_mask, attention_mask):
        '''
        Input:
            input_mask: [B, C*N]                   0 indicates the padding mask => not attend to 
            attention_mask: [B, 1, C*N, C*N]       indicates the attention mask by design => True means not attend to
        Output:
            final_mask: [B, 1, C*N, C*N]
        '''
        assert input_mask.shape[-1] == attention_mask.shape[-1] == attention_mask.shape[-2], "input_mask and attention_mask must have the same length"
        pad_mask_bool = (input_mask == 0)  # [B, L]
        
        query_pad_mask = pad_mask_bool.unsqueeze(2)  # [B, L, 1]
        key_pad_mask = pad_mask_bool.unsqueeze(1)    # [B, 1, L]
        combined_pad_mask = query_pad_mask | key_pad_mask  # [B, L, L]
        combined_pad_mask = combined_pad_mask.unsqueeze(1)  # [B, 1, L, L]
        final_mask = attention_mask | combined_pad_mask
        assert final_mask.shape == attention_mask.shape
        return final_mask
    
    @staticmethod
    def mask_patch_to_seq_with_special_tokens(input_mask):
        '''
        Input:
            input_mask: [B, C, N]
        Output:
            final_mask: [B, C*N+N+1]
        '''
        B, C, N = input_mask.shape
        input_mask = input_mask.reshape(B, -1) # [B, C*N]
        # Create a new mask with extra columns for special tokens
        final_length = C*N + C + 1  # Total length including special tokens
        final_mask = torch.zeros((B, final_length), device=input_mask.device, dtype=input_mask.dtype)
    
        final_mask[:, 0] = 1
        
        for t in range(C):
            special_token_pos = 1 + t*(N+1)  # Position for channel special token
            data_start = special_token_pos + 1  # Start position for channel data
            data_end = data_start + N  # End position for channel data
            final_mask[:, special_token_pos] = 1
            channel_data = input_mask[:, t*N:(t+1)*N]
            final_mask[:, data_start:data_end] = channel_data
            
        return final_mask
        
    
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        '''
        True means masked, False means observed => Tensor.masked_fill(MASK.mask, -inf)
        '''
        return self._mask


class TRACEMask():
    def __init__(self, B, n_vars, n_tokens, allow_cross_channel=True, device="cpu"):
        seq_length = 1 + n_vars + n_vars*n_tokens
        with torch.no_grad():           
            mask = torch.zeros((B, 1, seq_length, seq_length), dtype=torch.bool).to(device)
            if allow_cross_channel:
                for k in range(n_vars):
                    ch_ts_start = 1 + k*(n_tokens+1)
                    ch_ts_end = ch_ts_start + n_tokens+1
                    mask[:, 0, ch_ts_start, 0: ch_ts_start] = True
                    mask[:, 0, ch_ts_start, ch_ts_end:] = True
                    mask[:, 0, 0: ch_ts_start, ch_ts_start] = True
                    mask[:, 0, ch_ts_end:, ch_ts_start] = True
            else:
                for k in range(n_vars):
                    ch_ts_start = 1 + k*(n_tokens+1)
                    ch_ts_end = ch_ts_start + n_tokens+1
                    mask[:, 0, ch_ts_start:ch_ts_end, 0: ch_ts_start] = True
                    mask[:, 0, ch_ts_start:ch_ts_end, ch_ts_end:] = True
                    mask[:, 0, 0: ch_ts_start, ch_ts_start:ch_ts_end] = True
                    mask[:, 0, ch_ts_end:, ch_ts_start:ch_ts_end] = True
            mask[:, 0, 0, :] = False
            mask[:, 0, :, 0] = False
            self._mask = mask
    @property
    def mask(self):
        return self._mask
