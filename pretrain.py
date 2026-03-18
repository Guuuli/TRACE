import argparse
import os
import torch
import torch.distributed as dist

from src.common import PATHS
from src.tasks.pretrain_task import Pretraining
from src.utils.config import Config
from src.utils.tools import control_randomness, make_dir_if_not_exists, parse_config

def main_worker():
    # --------- Runtime Setup ----------
    local_rank = 0
    global_rank = 0
    world_size = 1

    # --------- Load Config ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--patch_stride_len", type=int, default=None)
    parser.add_argument("--seq_len_channel", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--pos_embed_type", type=str, default=None)
    args_cmd = parser.parse_args()

    config = Config(
        config_file_path=args_cmd.config,
        default_config_file_path="configs/default.yaml"
    ).parse()

    control_randomness(config["random_seed"])

    config["device"] = local_rank
    config["rank"] = global_rank
    config["world_size"] = world_size
    config["distributed"] = False
    config["checkpoint_path"] = PATHS.CHECKPOINTS_DIR
    ### Override config with command line arguments
    config["patch_len"] = args_cmd.patch_len
    config["patch_stride_len"] = args_cmd.patch_stride_len
    config["seq_len_channel"] = args_cmd.seq_len_channel
    config["train_batch_size"] = args_cmd.train_batch_size
    config["val_batch_size"] = args_cmd.val_batch_size
    config["pos_embed_type"] = args_cmd.pos_embed_type
    config["max_epochs"] = args_cmd.max_epochs
    make_dir_if_not_exists(config["checkpoint_path"])

    args = parse_config(config)

    print(f"[Rank {global_rank}] Running with config:\n{args}\n")
    task_obj = Pretraining(args=args)

    if global_rank == 0:
        task_obj.setup_logger(notes="Pre-training runs")

    task_obj.train()

    if global_rank == 0:
        task_obj.end_logger()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main_worker()
