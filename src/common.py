import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class TASKS:
    RECONSTRUCTION: str = "reconstruction"
    FORECASTING: str = "forecasting"
    CLASSIFICATION: str = "classification"
    EMBEDDING: str = "embedding"
    PRETRAINING: str = "pretraining"
    RAG: str = "rag"


def set_transformers_cache_path(transformers_cache_path: str):
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache_path


@dataclass
class PATHS:
    DATA_DIR: str = os.getenv("TTRAG_DATA_DIR")
    CHECKPOINTS_DIR: str = os.getenv("TTRAG_CHECKPOINTS_DIR")
    RESULTS_DIR: str = os.getenv("TTRAG_RESULTS_DIR")
    WANDB_DIR: str = os.getenv("WANDB_DIR")
    

EVENT_MAP = {'Lightning': 0, 
             'Debris Flow': 1, 
             'Flash Flood': 2, 
             'Heavy Rain': 3, 
             'Tornado': 4, 
             'Funnel Cloud': 5, 
             'Hail': 6, 
             'Flood': 7,
             'Thunderstorm Wind': 8,
             'None': -100}