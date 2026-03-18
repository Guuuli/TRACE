import os
import argparse
import torch
import torch.distributed as dist

from src.common import PATHS
from src.tasks.context_align_task import ContextAligning
from src.utils.config import Config
from src.utils.tools import control_randomness, make_dir_if_not_exists, parse_config
os.environ["TOKENIZERS_PARALLELISM"] = "false"

NOTES = "Aligning"

def main_worker():
    # ========== Runtime Setup ==========
    local_rank = 0
    global_rank = 0
    world_size = 1

    # ========== CLI Argument Parsing ==========
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/align.yaml")
    parser.add_argument("--pretraining_run_name", type=str, default="")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--finetuning_mode", type=str, default="end-to-end")
    parser.add_argument("--init_lr", type=float, default=0.001)
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--text_encoder_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_negatives", type=int, default=10)
    parser.add_argument('--hard_negative_mining', action='store_true', default=False)
    parser.add_argument("--cross_attend", action='store_true', default=False)
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
    config["distributed"] = False
    config["checkpoint_path"] = PATHS.CHECKPOINTS_DIR + "context_align/"
    config["result_dir"] = PATHS.RESULTS_DIR + "context_align/"
    make_dir_if_not_exists(config["checkpoint_path"])

    args = parse_config(config)
    args.pretraining_run_name = args_cmd.pretraining_run_name
    args.train_batch_size = args_cmd.train_batch_size
    args.val_batch_size = args_cmd.val_batch_size
    args.finetuning_mode = args_cmd.finetuning_mode
    args.max_epoch = args_cmd.max_epoch
    args.init_lr = args_cmd.init_lr
    args.text_encoder_name = args_cmd.text_encoder_name
    args.num_negatives = args_cmd.num_negatives
    args.hard_negative_mining = args_cmd.hard_negative_mining
    args.cross_attend = args_cmd.cross_attend
    if global_rank == 0:
        print(f"[Rank {global_rank}] Running with config:\n{args}\n")

    # ========== Main Training ==========
    task_obj = ContextAligning(args=args)
    if global_rank == 0:
        task_obj.setup_logger(notes=NOTES+f" on {args.pretraining_run_name}")

    task_obj.train()

    if global_rank == 0:
        task_obj.end_logger()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main_worker()
