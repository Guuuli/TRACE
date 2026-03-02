import os
import argparse
import torch
import torch.distributed as dist

from src.common import PATHS
from src.tasks.forecast_finetune_task import ForecastFinetuning
from src.utils.config import Config
from src.utils.tools import control_randomness, make_dir_if_not_exists, parse_config

NOTES = "Supervised finetuning on forecasting datasets"

def main_worker():
    # ========== Distributed Environment Setup ==========w
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # ========== CLI Argument Parsing ==========
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    parser.add_argument("--pretraining_run_name", type=str, default="")
    parser.add_argument("--forecast_horizon", type=int, default=24)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--finetuning_mode", type=str, default="linear-probing")
    parser.add_argument("--init_lr", type=float, default=0.001)
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--ts_only", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args_cmd = parser.parse_args()

    # ========== Config & Reproducibility ==========
    config = Config(
        config_file_path=args_cmd.config,
        default_config_file_path="configs/default.yaml"
    ).parse(run_name=args_cmd.pretraining_run_name, if_override=True)

    control_randomness(config["random_seed"])
    
    config["device"] = local_rank
    config["rank"] = global_rank
    config["world_size"] = world_size
    config["distributed"] = True
    config["checkpoint_path"] = PATHS.CHECKPOINTS_DIR + "forecasting/"
    make_dir_if_not_exists(config["checkpoint_path"])

    args = parse_config(config)
    
    args.task_name = "forecasting"
    args.pretraining_run_name = args_cmd.pretraining_run_name
    args.forecast_horizon = args_cmd.forecast_horizon
    args.train_batch_size = args_cmd.train_batch_size
    args.val_batch_size = args_cmd.val_batch_size
    args.finetuning_mode = args_cmd.finetuning_mode
    args.max_epoch = args_cmd.max_epoch
    args.init_lr = args_cmd.init_lr
    args.save_model = args_cmd.save_model
    args.top_k = args_cmd.top_k
    args.ts_only = args_cmd.ts_only
    args.debug = args_cmd.debug
    
    if global_rank == 0:
        print(f"[Rank {global_rank}] Running with config:\n{args}\n")

    # ========== Main Training ==========
    task_obj = ForecastFinetuning(args=args)
    if global_rank == 0:
        task_obj.setup_logger(notes="RAG Forecasting")

    task_obj.train()

    if global_rank == 0:
        task_obj.end_logger()

    dist.destroy_process_group()


if __name__ == "__main__":
    main_worker()