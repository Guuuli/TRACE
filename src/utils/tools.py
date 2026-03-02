import numpy as np
import torch
import matplotlib.pyplot as plt

import os
import random
from argparse import Namespace
from typing import NamedTuple
import torch.distributed as dist
import numpy as np
import torch

plt.switch_backend('agg')

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    return total, trainable

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    
    
class NamespaceWithDefaults(Namespace):
    @classmethod
    def from_namespace(cls, namespace):
        new_instance = cls()
        for attr in dir(namespace):
            if not attr.startswith("__"):
                setattr(new_instance, attr, getattr(namespace, attr))
        return new_instance

    def getattr(self, key, default=None):
        return getattr(self, key, default)


def parse_config(config: dict) -> NamespaceWithDefaults:
    args = NamespaceWithDefaults(**config)
    return args


def make_dir_if_not_exists(path, verbose=True):
    if not is_directory(path):
        path = path.split(".")[0]
    if not os.path.exists(path=path):
        os.makedirs(path)
        if verbose:
            print(f"Making directory: {path}...")
    return True


def is_directory(path):
    extensions = [".pth", ".txt", ".json", ".yaml"]

    for ext in extensions:
        if ext in path:
            return False
    return True


def control_randomness(seed: int = 13):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricsStore(NamedTuple):
    train_loss: dict = None
    val_loss: dict = None
    test_loss: dict = None


def dtype_map(dtype: str):
    map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    return map[dtype]


def get_huggingface_model_dimensions(model_name: str = "flan-t5-base"):
    from transformers import T5Config

    config = T5Config.from_pretrained(model_name)
    return config.d_model


def get_anomaly_criterion(anomaly_criterion: str = "mse"):
    if anomaly_criterion == "mse":
        return torch.nn.MSELoss(reduction="none")
    elif anomaly_criterion == "mae":
        return torch.nn.L1Loss(reduction="none")
    else:
        raise ValueError(f"Anomaly criterion {anomaly_criterion} not supported.")


def _reduce(metric, reduction="mean", axis=None):
    if reduction == "mean":
        return np.nanmean(metric, axis=axis)
    elif reduction == "sum":
        return np.nansum(metric, axis=axis)
    elif reduction == "none":
        return metric


class EarlyStopping:
    def __init__(self, patience=7, delta=0, verbose=True, mode='min'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.best_loss = float("inf") if mode == "min" else -float("inf")

    def __call__(self, val_metric, model, path):
        score = -val_metric if self.mode == "min" else val_metric

        if self.best_score is None or \
           (self.mode == "min" and score < self.best_score - self.delta) or \
           (self.mode == "max" and score > self.best_score + self.delta):
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)
            
import torch.nn as nn

class MultiHeadWrapper(nn.Module):
    def __init__(self, heads: dict[str, nn.Module]):
        super().__init__()
        for name, module in heads.items():
            self.add_module(name, module)

        self._head_names = list(heads.keys())  # for later access

    def forward(self, *args, **kwargs):
        # Optional: raise error if directly called
        raise NotImplementedError("Call individual heads like model.head.reconstruct_head(x)")

    def keys(self):
        return self._head_names

    def __getitem__(self, name):
        return getattr(self, name)



def broadcast_string(s, src=0):
    import torch
    import torch.distributed as dist

    max_len = 256 
    tensor = torch.zeros(max_len, dtype=torch.uint8, device="cuda")

    if dist.get_rank() == src:
        encoded = s.encode("utf-8")
        tensor[:len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8, device="cuda")

    dist.broadcast(tensor, src=src)

    decoded = bytes([x for x in tensor.tolist() if x > 0]).decode("utf-8")
    return decoded


def gather_across_gpus(obj):
    """Gather list of python objects across all ranks."""
    world_size = dist.get_world_size()
    gathered_obj = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_obj, obj)
    return gathered_obj

def flatten_nested_list(nested):
    return [item for sublist in nested for item in sublist]



def gather_all_tensor(tensor):
    # gather tensor from all GPUs
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)


def gather_all_tensor_with_padding(tensor, pad_value=0.0):
    """
    Gather tensors of different lengths from all ranks and concatenate them along dim=0.
    - tensor: [N_local, ...] (can be 1D, 2D, etc.)
    - returns: [N_total, ...]
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    local_shape = torch.tensor([tensor.shape[0]], device=tensor.device)
    
    # Step 1: Gather all lengths
    all_lengths = [torch.zeros_like(local_shape) for _ in range(world_size)]
    dist.all_gather(all_lengths, local_shape)
    all_lengths = [l.item() for l in all_lengths]
    max_len = max(all_lengths)

    # Step 2: Pad local tensor if needed
    if tensor.shape[0] < max_len:
        pad_size = [max_len - tensor.shape[0]] + list(tensor.shape[1:])
        pad_tensor = torch.full(pad_size, pad_value, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad_tensor], dim=0)

    # Step 3: All gather padded tensors
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    # Step 4: Truncate each gathered tensor to its actual length
    trimmed = [g[:l] for g, l in zip(gathered, all_lengths)]

    # Step 5: Concatenate all
    return torch.cat(trimmed, dim=0)


def gather_all_list_strings(all_raw_descriptions):
    if dist.get_world_size() > 1:
        local_raw_descriptions = all_raw_descriptions  # list[str]
        gathered_raw_descriptions = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_raw_descriptions, local_raw_descriptions)

        # Flatten list of lists
        all_raw_descriptions = sum(gathered_raw_descriptions, [])
    else:
        all_raw_descriptions = all_raw_descriptions
    return all_raw_descriptions