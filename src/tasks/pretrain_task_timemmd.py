import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.tools import MetricsStore, dtype_map, make_dir_if_not_exists, count_parameters
from .base import Tasks
from torch.utils.data.distributed import DistributedSampler
warnings.filterwarnings("ignore")


class PretrainingTimeMMD(Tasks):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        self._build_model()
        count_parameters(self.model)
    
    def validation(self, data_loader, split: str = "val"):
        loss = {f"{split}_mse": [], f"{split}_mae": []}

        self.model.eval()
        with torch.no_grad():
            for batch_x in tqdm(data_loader, total=len(data_loader)):
                timeseries = batch_x.timeseries.float().to(self.device)  # [B, C, L]
                input_mask = batch_x.input_mask.long().to(self.device)  # [B, C, L]
                prior_y = batch_x.prior_y.float().to(self.device)  # [B, 1, H]
                forecast_label = batch_x.forecast.float().to(self.device)[:, :, -self.args.forecast_horizon:]  # [B, 1, H]

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=None
                    )

                forecast_output = (1 - self.args.prior_ratio) * outputs.forecast + self.args.prior_ratio * prior_y
                forecast_output = forecast_output.detach().cpu().numpy().reshape(-1, self.args.forecast_horizon)
                forecast_label = forecast_label.detach().cpu().numpy().reshape(-1, self.args.forecast_horizon)
                
                mse = np.mean((forecast_output - forecast_label) ** 2)
                mae = np.mean(np.abs(forecast_output - forecast_label))

                loss[f"{split}_mse"].append(mse)
                loss[f"{split}_mae"].append(mae)

        # Aggregate across all processes if using DDP
        avg_mse = torch.tensor(np.mean(loss[f"{split}_mse"]), device=self.device)
        avg_mae = torch.tensor(np.mean(loss[f"{split}_mae"]), device=self.device)

        if self.args.world_size > 1:
            torch.distributed.all_reduce(avg_mse, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(avg_mae, op=torch.distributed.ReduceOp.SUM)
            avg_mse /= self.args.world_size
            avg_mae /= self.args.world_size

        return {f"{split}_mse": avg_mse.item(), f"{split}_mae": avg_mae.item()}

    def train(self):
        if self.args.rank == 0:
            self.run_name = self.logger.name
            path = os.path.join(self.args.checkpoint_path, self.run_name)
            make_dir_if_not_exists(path, verbose=True)

        self.optimizer = self._select_optimizer()
        self.forecast_criterion = self._select_criterion()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self._init_lr_scheduler()
        self.model.to(self.args.rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.rank], find_unused_parameters=True)

        best_mse = float("inf")  ### EARLY STOP ###
        patience_counter = 0     ### EARLY STOP ###

        opt_steps = 0
        cur_epoch = 0
        while cur_epoch < self.args.max_epoch:
            print(f"Epoch {cur_epoch} of {self.args.max_epoch}")
            self.model.train()
            if self.args.distributed and isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(cur_epoch)

            for batch_x in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
                self.optimizer.zero_grad(set_to_none=True)
                timeseries = batch_x.timeseries.float().to(self.device)  # [B, 1, L]
                input_mask = batch_x.input_mask.long().to(self.device)   # [B, 1, L]
                prior_y = batch_x.prior_y.float().to(self.device)        # [B, 1, H]
                forecast_label = batch_x.forecast.float().to(self.device)[:, :, -self.args.forecast_horizon:]

                if not self.args.set_input_mask:
                    input_mask = torch.ones_like(input_mask)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(x_enc=timeseries, input_mask=input_mask)

                recon_loss = self.forecast_criterion(outputs.reconstruction, timeseries)
                observed_mask = input_mask * (1 - outputs.pretrain_mask)
                masked_loss = observed_mask * recon_loss
                recon_loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

                forecast_output = (1 - self.args.prior_ratio) * outputs.forecast + self.args.prior_ratio * prior_y
                forecast_loss = self.forecast_criterion(forecast_output, forecast_label)
                total_loss = recon_loss + forecast_loss

                if self.args.rank == 0 and opt_steps % 100 == 0:
                    self.logger.log({
                        "train_total_loss": total_loss.item(),
                        "train_recon_loss": recon_loss.item(),
                        "train_forecast_loss": forecast_loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    })

                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                opt_steps += 1
                self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)

            # evaluation & early stop
            if cur_epoch % self.args.log_interval == 0:
                if self.args.distributed and isinstance(self.val_dataloader.sampler, DistributedSampler):
                    self.val_dataloader.sampler.set_epoch(cur_epoch)

                eval_metrics = self.evaluate_and_log()
                test_mse = eval_metrics.test_loss.get("test_mse", None)

                ### EARLY STOP ###
                if test_mse is not None:
                    if test_mse < best_mse - self.args.delta:
                        best_mse = test_mse
                        patience_counter = 0
                        if self.args.rank == 0:
                            best_checkpoint_path = os.path.join(self.args.checkpoint_path, f"{self.run_name}_best_checkpoint.pt")
                            torch.save({
                                'model_state_dict': self.model.state_dict(),
                                'args': self.args
                            }, best_checkpoint_path)
                            print(f"Saved new best checkpoint to {best_checkpoint_path}")
                    else:
                        patience_counter += 1
                        print(f"Patience counter: {patience_counter}/{self.args.patience}")
                        if patience_counter >= self.args.patience:
                            print(f"Early stopping at epoch {cur_epoch}")
                            return self.model
                ### EARLY STOP ###

            cur_epoch += 1

        return self.model

    def evaluate_and_log(self):
        eval_metrics = MetricsStore(test_loss=self.validation(self.test_dataloader, split="test"))
        if self.args.rank == 0:
            self.logger.log(eval_metrics.test_loss)
        return eval_metrics

