import torch



def decompose_token_sequence(token_sequence: torch.Tensor, num_channels: int):
    """
    Decompose the token sequence into the original sequence
    Input: 
        token_sequence: torch.Tensor of shape [B, Seq_len, d_model] with Seq_len = C * L + C + 1
    Output:
        x: torch.Tensor of shape [B, C, L, d_model]
        channel_tokens: torch.Tensor of shape [B,C, d_model]
        cls_token: torch.Tensor of shape [B, d_model]
    """
    B, seq_len, d_model = token_sequence.shape
    assert (seq_len-1) % num_channels == 0, "The number of channels must be divisible by the length of the token sequence"
    num_patches = (seq_len-1) // num_channels - 1
    timeseries_tokens = []
    channels_tokens = []
    for i in range(num_channels):
        start_idx = 2 + i * (num_patches + 1)
        channels_tokens.append(token_sequence[:, start_idx-1,:])
        timeseries_tokens.append(token_sequence[:, start_idx:start_idx+num_patches,:])
    cls_token = token_sequence[:, 0, :] #[B, d_model]
    channels_tokens = torch.stack(channels_tokens, dim=1) #[B, C, d_model]
    x = torch.stack(timeseries_tokens, dim=1) #[B, C, L, d_model]
    return x, channels_tokens, cls_token


