from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy.typing as npt
import torch
from torch.utils.data import Dataset


@dataclass
class TimeseriesData:
    timeseries: npt.NDArray = None
    forecast: npt.NDArray = None
    labels: Union[npt.NDArray, int, str] = None
    input_mask: npt.NDArray = None
    metadata: dict = None
    name: str = None
    description_emb: torch.Tensor = None
    event_emb: torch.Tensor = None
    channel_description_emb: torch.Tensor = None
    descriptions: List[str] = None
    channel_descriptions: List[List[str]] = None
    events: List[List[str]] = None
    prior_y: npt.NDArray = None


@dataclass
class TimeseriesOutputs:
    forecast: npt.NDArray = None
    labels: int = None
    input_mask: npt.NDArray = None
    pretrain_mask: npt.NDArray = None
    reconstruction: npt.NDArray = None
    embeddings: npt.NDArray = None
    channel_embeddings: npt.NDArray = None
    cls_embedding: npt.NDArray = None
    metadata: dict = None
    illegal_output: bool = False
    classification: npt.NDArray = None
    description_emb: torch.Tensor = None
    event_emb: torch.Tensor = None
    channel_description_emb: torch.Tensor = None




@dataclass
class DataSplits:
    train: npt.NDArray = None
    val: npt.NDArray = None
    test: npt.NDArray = None


@dataclass
class ClassificationResults:
    train_embeddings: npt.NDArray = None
    test_embeddings: npt.NDArray = None
    train_labels: npt.NDArray = None
    test_labels: npt.NDArray = None
    train_predictions: npt.NDArray = None
    test_predictions: npt.NDArray = None
    train_accuracy: float = None
    test_accuracy: float = None
    dataset_name: str = None





class TaskDataset(ABC, Dataset):
    def __init__(self):
        super(TaskDataset, self).__init__()

    def _read_data(self) -> TimeseriesData:
        return NotImplementedError

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):
        return NotImplementedError

    def plot(self, idx):
        return NotImplementedError

    def _check_and_remove_nans(self):
        return NotImplementedError

    def _subsample(self):
        return NotImplementedError