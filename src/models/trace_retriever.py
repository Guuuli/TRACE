from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.common import PATHS
from src.data.dataloader import get_dataloader
from src.models.mm_encoder import MultiModalEncoder


class RetrievalAugmentedWrapper(nn.Module):
    def __init__(
        self,
        device,
        checkpoint_path: str | "results/model_checkpoints/context_align/retriever_demo.pt" = None,
        retrieval_split: str = "train",
        embedding_dir: str | None = None,
        batch_size: int = 256,
    ):
        super().__init__()
        self.device = device
        self.retrieval_split = retrieval_split
        self.batch_size = batch_size

        self.checkpoint_path = checkpoint_path
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model_args = self._normalize_args(checkpoint.get("args"))

        self.encoder = MultiModalEncoder(model_args)
        new_state_dict = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in checkpoint["model_state_dict"].items()
        }
        self.encoder.load_state_dict(new_state_dict)
        self.encoder.to(self.device)
        self.encoder.eval()

        if embedding_dir:
            self.embedding_dir = Path(embedding_dir)
        else:
            data_root = Path(PATHS.DATA_DIR) if PATHS.DATA_DIR else Path("dataset")
            self.embedding_dir = data_root / "retrieval"
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

        self.ts_embedding_path = self.embedding_dir / "ts_embedding.npy"
        self.text_embedding_path = self.embedding_dir / "text_embedding.npy"
        self.timeseries_path = self.embedding_dir / "timeseries.npy"

        if not self._embedding_bank_exists():
            self._build_embedding_bank(model_args)

        self.text_emb = torch.tensor(np.load(self.text_embedding_path), dtype=torch.float32).to(self.device)
        self.ts_emb = torch.tensor(np.load(self.ts_embedding_path), dtype=torch.float32).to(self.device)
        self.timeseries = torch.tensor(np.load(self.timeseries_path), dtype=torch.float32).to(self.device)
        self.num_channels = self.timeseries.shape[1]

        self.text_emb = F.normalize(self.text_emb, dim=-1)
        self.ts_emb = F.normalize(self.ts_emb, dim=-1)
        self.ts_embedding_dim = self.ts_emb.shape[-1]

    def _normalize_args(self, args):
        if isinstance(args, dict):
            args = Namespace(**args)
        if not hasattr(args, "rank"):
            args.rank = 0
        if not hasattr(args, "finetuning_mode"):
            args.finetuning_mode = "end-to-end"
        if not hasattr(args, "cross_attend"):
            args.cross_attend = False
        if not hasattr(args, "model_name"):
            args.model_name = "TraceEncoder"
        return args


    def _embedding_bank_exists(self) -> bool:
        return (
            self.ts_embedding_path.exists()
            and self.text_embedding_path.exists()
            and self.timeseries_path.exists()
        )

    def _build_embedding_bank(self, model_args):
        print(f"[Retriever] Building embedding bank from split='{self.retrieval_split}'")

        model_args.task_name = "retrieval"
        model_args.data_split = self.retrieval_split
        model_args.batch_size = self.batch_size
        model_args.train_batch_size = self.batch_size
        model_args.val_batch_size = self.batch_size
        model_args.device = self.device
        model_args.distributed = False

        data_loader = get_dataloader(model_args)

        all_timeseries = []
        all_ts_embeddings = []
        all_text_embeddings = []

        self.encoder.eval()
        with torch.no_grad():
            for batch_x in tqdm(data_loader, total=len(data_loader), desc="Encoding retrieval bank"):
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device)

                if hasattr(model_args, "set_input_mask") and not model_args.set_input_mask:
                    input_mask = torch.ones_like(input_mask)

                outputs = self.encoder(
                    x_enc=timeseries,
                    input_mask=input_mask,
                    channel_description_emb=batch_x.channel_description_emb.to(self.device),
                    description_emb=batch_x.description_emb.to(self.device),
                    event_emb=batch_x.event_emb.to(self.device),
                )

                all_timeseries.append(timeseries.detach().cpu())
                all_ts_embeddings.append(outputs.embeddings.detach().cpu())
                all_text_embeddings.append(outputs.description_emb.detach().cpu())

        ts_embeddings = F.normalize(torch.cat(all_ts_embeddings, dim=0), dim=-1).numpy()
        text_embeddings = F.normalize(torch.cat(all_text_embeddings, dim=0), dim=-1).numpy()
        timeseries = torch.cat(all_timeseries, dim=0).numpy()

        np.save(self.ts_embedding_path, ts_embeddings)
        np.save(self.text_embedding_path, text_embeddings)
        np.save(self.timeseries_path, timeseries)
        print(f"[Retriever] Saved bank to {self.embedding_dir}")

    def forward(self, x_enc, input_mask, top_k=5):
        with torch.no_grad():
            ts_query_out = self.encoder.get_ts_embedding(x_enc, input_mask)
            ts_query_emb = F.normalize(ts_query_out.embeddings, dim=-1)

            ts_sim = ts_query_emb @ self.ts_emb.T
            _, ts_topk_idx = torch.topk(ts_sim, top_k, dim=-1)
            flat_idx = ts_topk_idx.view(-1)

            text_topk = self.text_emb[flat_idx].view(ts_topk_idx.shape[0], top_k, -1)
            ts_topk = self.ts_emb[flat_idx].view(ts_topk_idx.shape[0], top_k, -1)
            timeseries_topk = self.timeseries[flat_idx].view(ts_topk_idx.shape[0], top_k, self.num_channels, -1)

            return {
                "text_topk": text_topk,
                "ts_topk": ts_topk,
                "timeseries_topk": timeseries_topk,
            }
