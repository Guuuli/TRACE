import os
import warnings
from copy import deepcopy

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch import optim
from wandb import AlertLevel

from src.common import PATHS
from src.data.dataloader import get_dataloader
from src.models.timeseries_encoders.base import BaseModel
from src.utils.optims import LinearWarmupCosineLRScheduler
from src.utils.tools import MetricsStore
from src.utils.metrics import sMAPELoss
from src.models.timeseries_encoders.ts_encoder import TS_Encoder
from src.models.timeseries_encoders.dLinear import DLinear
from torch.nn.parallel import DistributedDataParallel
from src.models.mm_encoder import MultiModalEncoder
warnings.filterwarnings("ignore")


class Tasks(nn.Module):
    '''
    Base task class for Timeseries Encoder (Pre-training and Finetuning).
    Require overiding for Timeseries-Text Encoder (Alignment):
        - _build_model
        - _select_criterion
        - _load_pretrained_ts_encoder
        etc.
    '''
    def __init__(self, args, **kwargs):
        super(Tasks, self).__init__()
        self.args = args
        self._dataloader = {}
        
        self._acquire_device()
        if self.args.task_name != "retrieval":
            self._build_model()

        # Setup data loaders
        self.train_dataloader = self._get_dataloader(data_split="train")
        self.test_dataloader = self._get_dataloader(data_split="test")
        self.val_dataloader = self._get_dataloader(data_split="val") if self.args.task_name != "retrieval" else None

    def _build_model(self):
        if self.args.model_name in ["TraceEncoder"]:
            self.model = TS_Encoder(configs=self.args)
        elif self.args.model_name == "DLinear":
            self.model = DLinear(configs=self.args)
        else:
            raise NotImplementedError(f"Model {self.args.model_name} not implemented")
        return self.model

    def _acquire_device(self):
        self.device = torch.device(f"cuda:{self.args.rank}" if torch.cuda.is_available() else "cpu")
        return self.device

    def _reset_dataloader(self):
        self._dataloader = {}

    def _get_dataloader(self, data_split: str = "train"):
        # Load Datasets
        if self._dataloader.get(data_split) is not None:
            return self._dataloader.get(data_split)
        else:
            data_loader_args = deepcopy(self.args)
            data_loader_args.data_split = data_split
            data_loader_args.batch_size = (
                self.args.train_batch_size
                if data_split == "train"
                else self.args.val_batch_size
            )
            data_loader_args.shuffle = False if data_split == "test" else True
            print(f"Loading {data_split} split of the dataset")

            self._dataloader[data_split] = get_dataloader(
                args=data_loader_args
            )
            return self._dataloader.get(data_split)

    def _select_optimizer(self):
        if self.args.optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.init_lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer_name == "Adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.init_lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer_name == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.init_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.args.optimizer_name} not implemented"
            )
        return optimizer

    def _init_lr_scheduler(self, type: str = "linearwarmupcosinelr"):
        decay_rate = self.args.lr_decay_rate
        warmup_start_lr = self.args.warmup_lr
        warmup_steps = self.args.warmup_steps

        if type == "linearwarmupcosinelr":
            self.lr_scheduler = LinearWarmupCosineLRScheduler(
                optimizer=self.optimizer,
                max_epoch=self.args.max_epoch,
                min_lr=self.args.min_lr,
                init_lr=self.args.init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )
        elif type == "onecyclelr":
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.args.init_lr,
                epochs=self.args.max_epoch,
                steps_per_epoch=len(self.train_dataloader),
                pct_start=self.args.pct_start,
            )
        elif type == "none":
            self.lr_scheduler = None

    def _select_criterion(
        self, loss_type: str = "mse", reduction: str = "mean", **kwargs
    ):
        if loss_type == "mse":
            criterion = nn.MSELoss(reduction=reduction)
        elif loss_type == "mae":
            criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == "huber":
            criterion = nn.HuberLoss(reduction=reduction, delta=kwargs["delta"])
        elif loss_type == "smape":
            criterion = sMAPELoss(reduction=reduction)
        return criterion

    def save_results(self, results_df: pd.DataFrame, path: str):
        results_df.to_csv(
            os.path.join(path, f"results_{self.args.task_name}.csv")
        )

    def save_model(
        self,
        model: nn.Module,
        path: str,
        opt_steps: int,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
    ):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }

        if opt_steps is None:
            with open(os.path.join(path, f"{self.args.model_name}.pth"), "wb") as f:
                torch.save(checkpoint, f)
        else:
            with open(os.path.join(path, f"{self.args.model_name}_checkpoint_{opt_steps}.pth"), "wb") as f:
                torch.save(checkpoint, f)

    def save_model_and_alert(self, opt_steps):
        self.save_model(
            self.model, self.checkpoint_path, opt_steps, self.optimizer, self.scaler
        )

    def load_pretrained_ts_encoder(
        self, pretraining_task_name: str = "pretraining", do_not_copy_head: bool = True
    ):
        pretraining_args = deepcopy(self.args)
        pretraining_args.task_name = pretraining_task_name

        checkpoint = BaseModel.load_pretrained_weights(
            run_name=pretraining_args.pretraining_run_name,
            opt_steps=pretraining_args.pretraining_opt_steps,
            model_name=pretraining_args.model_name,
        )
        if self.args.model_name in ["TraceEncoder", "TimerXL", "MOMENT"]:
            pretrained_model = TS_Encoder(configs=pretraining_args)
        elif self.args.model_name == "DLinear":
            pretrained_model = DLinear(configs=pretraining_args)
        else:
            raise NotImplementedError(f"Model {self.args.model_name} not implemented")
        
        new_state_dict = {}
        for k, v in checkpoint["model_state_dict"].items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        pretrained_model.load_state_dict(new_state_dict)

        # Copy pre-trained parameters to fine-tuned model
        for (name_p, param_p), (name_f, param_f) in zip(
            pretrained_model.named_parameters(), self.model.named_parameters()
        ):
            if (name_p == name_f) and (param_p.shape == param_f.shape):
                if do_not_copy_head and name_p.startswith("head"):
                    continue
                else:
                    param_f.data = param_p.data

        self.freeze_model_parameters()  # Freeze model parameters based on fine-tuning mode

        return True

    def freeze_model_parameters(self):
        if self.args.finetuning_mode == "linear-probing":
            for name, param in self.model.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad = False
        elif self.args.finetuning_mode == "end-to-end":
            pass
        else:
            raise NotImplementedError(
                f"Finetuning mode {self.args.finetuning_mode} not implemented"
            )

        print("====== Frozen parameter status ======")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("Not frozen:", name)
            else:
                print("Frozen:", name)
        print("=====================================")

    def _create_results_dir(self, experiment_name="supervised_forecasting"):
        if experiment_name == "supervised_forecasting":
            results_path = os.path.join(
                PATHS.RESULTS_DIR,
                experiment_name,
                self.args.model_name,
                self.args.finetuning_mode,
                self.args.pretraining_run_name,
                str(self.args.forecast_horizon),
            )
        elif experiment_name == "retrieval":
            results_path = os.path.join(
                PATHS.RESULTS_DIR,
                experiment_name,
                self.args.model_name,
                self.args.pretraining_run_name,
            )
        os.makedirs(results_path, exist_ok=True)
        return results_path

    def setup_logger(self, notes: str = None):
        self.logger = wandb.init(
            project="Timeseries Multimodal Foundation Model",
            dir=PATHS.WANDB_DIR,
            config=self.args,
            name=self.args.run_name if hasattr(self.args, "run_name") else None,
            notes=self.args.notes if notes is None else notes,
            mode="disabled" if self.args.debug else "run",
        )
        if self.args.debug:
            print(f"Run name: {self.logger.name}\n")
        return self.logger.name

    def end_logger(self):
        self.logger.finish()

    def evaluate_model(self):
        return MetricsStore(
            train_loss=self.validation(self.train_dataloader),
            test_loss=self.validation(self.test_dataloader),
            val_loss=self.validation(self.val_dataloader),
        )

    def evaluate_and_log(self):
        eval_metrics = self.evaluate_model()
        self.logger.log(
            {
                "train_loss": eval_metrics.train_loss,
                "validation_loss": eval_metrics.val_loss,
                "test_loss": eval_metrics.test_loss,
            }
        )
        return eval_metrics

    def debug_model_outputs(self, loss, outputs, batch_x, **kwargs):
        # Debugging code
        if (
            torch.any(torch.isnan(loss))
            or torch.any(torch.isinf(loss))
            or (loss < 1e-3)
        ):
            self.logger.alert(
                title="Loss is NaN or Inf or too small",
                text=f"Loss is {loss.item()}.",
                level=AlertLevel.INFO,
            )
            print("loss is NaN or Inf or too small")
            breakpoint()

        # Check model outputs
        if outputs.illegal_output:
            self.logger.alert(
                title="Model weights are NaN or Inf",
                text=f"Model weights are NaN or Inf.",
                level=AlertLevel.INFO,
            )
            print("model weights are NaN or Inf")
            breakpoint()

        # Check model gradients
        illegal_encoder_grads = (
            torch.stack(
                [torch.isfinite(p).any() for p in self.model.encoder.parameters()]
            )
            .any()
            .item()
        )
        illegal_head_grads = (
            torch.stack([torch.isfinite(p).any() for p in self.model.head.parameters()])
            .any()
            .item()
        )
        illegal_patch_embedding_grads = (
            torch.stack(
                [
                    torch.isfinite(p).any()
                    for p in self.model.patch_embedding.parameters()
                ]
            )
            .any()
            .item()
        )

        illegal_grads = (
            illegal_encoder_grads or illegal_head_grads or illegal_patch_embedding_grads
        )

        if illegal_grads:
            # self.logger.alert(title="Model gradients are NaN or Inf",
            #                     text=f"Model gradients are NaN or Inf.",
            #                     level=AlertLevel.INFO)
            # breakpoint()
            print("Model gradients are NaN or Inf.")
        if illegal_encoder_grads:
            print("Encoder gradients are NaN or Inf.")
        if illegal_head_grads:
            print("Head gradients are NaN or Inf.")
        if illegal_patch_embedding_grads:
            print("Patch embedding gradients are NaN or Inf.")

        return