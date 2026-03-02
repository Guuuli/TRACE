import torch
import torch.nn as nn
from src.utils.convert import decompose_token_sequence

class ReconstructionHead(nn.Module):
    def __init__(
        self,
        num_channels: int,
        d_model: int = 768,
        patch_len: int = 8,
        head_dropout: float = 0.1,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.dropout = nn.Dropout(head_dropout)
        self.patch_len = patch_len if patch_len > 0 else 1
        self.linear = nn.Linear(d_model, self.patch_len)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x, shape: str = "BTD"):
        """
        x: [B, total_len, d_model], where total_len = N * C + C + 1
            or [B, C, N, d_model]
        output: [B, C, L], where L = N * patch_len
        
        Note: reconstruction targets are on the point-wise level on multiple channels
        """
        if shape == "BTD":
            x, _, _ = decompose_token_sequence(x, self.num_channels)  #[B, C, N, d_model]
        x = self.linear(
            self.dropout(x)
        )  # [B, C, N, patch_len]
        x = x.flatten(start_dim=2, end_dim=3)  # [B, C, N * patch_len] => [B, C, L]
        return x


class EmbeddingHead(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
    def forward(self, x, input_mask_patch_view: torch.Tensor = None, shape: str = "BTD"):
        '''
        input:
            x: [B, total_len, d_model], where total_len = N * C + C + 1
                or [B, C, N, d_model]
            input_mask_patch_view: [B, C, N]
        output: channel-wise and global embeddings
        '''
        if shape == "BTD":
            x, channels_tokens, cls_token = decompose_token_sequence(x, num_channels=self.num_channels)  #[B, C, N, d_model]
        else:
            channels_tokens = x.mean(dim=2, keepdim=False)  #[B,C,d_model]
            cls_token = None
        x = x.mean(dim=1, keepdim=False)  # Mean across channels [B, N, d_model]    
        input_mask_patch_view = input_mask_patch_view[:,0,:].unsqueeze(-1).repeat(1,1,x.shape[-1])
        x = (input_mask_patch_view * x).sum(dim=1) / input_mask_patch_view.sum(dim=1)
        return {"channels": channels_tokens, "cls": cls_token, "global": x}


class ClassificationHead(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 768,
        n_classes: int = 2,
        head_dropout: int = 0.1,
        view: str = "global"
    ):
        super().__init__()
        self.emb_layer = EmbeddingHead(num_channels=n_channels)
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.view = view
        if view == "channels":
            self.linear = nn.Linear(n_channels * d_model, n_classes)
        else:
            self.linear = nn.Linear(d_model, n_classes)
    def forward(self, x, input_mask_patch_view: torch.Tensor = None, shape: str = "BTD"):
        """
        Input:
            x: [B, total_len, d_model], where total_len = N * C + C + 1
                or [B, C, N, d_model]
            input_mask_patch_view: [B, C, N]
        Output:
            [B, n_classes] if view == "global" or "cls" else [B, C, n_classes]
        """
        if self.view == "global":
            x = self.emb_layer(x, input_mask_patch_view, shape)["global"]  #[B, d_model]
        elif self.view == "channels":
            x = self.emb_layer(x, input_mask_patch_view, shape)["channels"]  #[B, C, d_model]
            x = self.flatten(x)  #[B, C * d_model]
        elif self.view == "cls":
            x = self.emb_layer(x, input_mask_patch_view, shape)["cls"]  #[B, d_model]
        x = self.dropout(x)
        y = self.linear(x)  
        return y


class ForecastingHead(nn.Module):
    def __init__(
        self, num_channels: int, d_model: int, num_patches: int, forecast_horizon: int = 96, head_dropout: int = 0
    ):
        super().__init__()
        self.num_channels = num_channels    
        self.head_nf = d_model * num_patches
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.forecast_horizon = forecast_horizon
        self.linear = nn.Linear(self.head_nf, forecast_horizon)

    def forward(self, x, input_mask: torch.Tensor = None, shape: str = "BTD"):
        """
        Input:
            x: [B, total_len, d_model], where total_len = N * C + C + 1
                or [B, C, N, d_model]
            input_mask: [B, C, N]
        Output:
            [B, C, forecast_horizon]
        """
        if shape == "BTD":
            x, _, _ = decompose_token_sequence(x, num_channels=self.num_channels)  #[B, C, N, d_model]
        x = self.flatten(x)  # x: [B, C, N * d_model] 
        x = self.linear(x)  # x: [B, C, forecast_horizon]
        x = self.dropout(x)  # x: [B, C, forecast_horizon]
        return x
    


class RetrievalAugmentedHead(nn.Module):
    def __init__(
        self, num_channels: int, d_model: int, num_patches: int, forecast_horizon: int = 96, head_dropout: int = 0, top_k: int = 3, ts_only: bool = False
    ):
        super().__init__()
        self.num_channels = num_channels    
        self.head_nf = d_model * num_patches
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.forecast_horizon = forecast_horizon
        self.ts_only = ts_only
        self.top_k = top_k
        if ts_only == True:
            self.text_token_layer = None
            self.forecast_head = nn.Linear(self.head_nf + d_model, forecast_horizon)
        else:
            self.text_token_layer = nn.Linear(d_model*self.top_k, d_model)
            self.forecast_head = nn.Linear(self.head_nf + d_model*2, forecast_horizon)
        self.ts_token_layer = nn.Linear(self.top_k*186, d_model)
        
    def forward(self, x, soft_prompt, shape: str = "BTD"):
        """
        Input:
            x: [B, total_len, d_model], where total_len = N * C + C + 1
                or [B, C, N, d_model]
            soft_prompt: dict
        Output:
            [B, C, forecast_horizon]
        """
        if shape == "BTD":
            x, _, _ = decompose_token_sequence(x, num_channels=self.num_channels)  #[B, C, N, d_model]
        x = self.flatten(x)  # x: [B, C, N * d_model] 
        B,C = x.shape[0], x.shape[1]
        top_text, top_ts = soft_prompt["text_topk"], soft_prompt["ts_topk"]  # [B,K,d]
        top_timeseries = soft_prompt["timeseries_topk"]  # [B,K,C,L]
        top_text = top_text.reshape(B, -1)   # [B, K*d]
        top_ts = top_ts.reshape(B, -1)   # [B, K*d]
        if self.ts_only != True:
            text_prompt = self.text_token_layer(top_text)  # [B, d]
            text_prompt = text_prompt.unsqueeze(1).expand(B, C, -1).contiguous()  # [B, C, d]
        B, K, C, L = top_timeseries.shape
        top_timeseries = top_timeseries.permute(0, 2, 1, 3)   # [B, C, K, L]
        top_timeseries = top_timeseries.reshape(B * C, K * L) # [B*C, K*L]
        top_timeseries = self.ts_token_layer(top_timeseries).reshape(B,C,-1)  #[B, C, d]
        if self.ts_only != True:
            soft_token = torch.cat([text_prompt, top_timeseries], dim=-1)  #[B, C, 2d]
        else:
            soft_token = top_timeseries  #[B, C, d]
        x = torch.cat([soft_token, x], dim=-1)  #[B, C, Nd+d or Nd+2d]
        x = self.forecast_head(x)  # x: [B, C, forecast_horizon]
        x = self.dropout(x)  # x: [B, C, forecast_horizon]
        return x