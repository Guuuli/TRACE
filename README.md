<h1><b>
(NeurIPS'25) TRACE: Grounding Time Series in Context for Multimodal Embedding and Retrieval
</b></h1>

**This repository provides installation and usage scripts for TRACE <a href="https://arxiv.org/abs/2506.09114" target="_blank">(arXiv:2506.09114)</a>.**


## 1) Create Environment

```bash
conda env create -f environment.yml
conda activate trace-rag
```

Configure runtime paths via `.env`.

## 2) Download Dataset

Download dataset from [Google Drive](https://drive.google.com/file/d/1hX4D91QbXa0UQlgf6Jnf-1ii96gfp1aY/view?usp=sharing) and unzip the file into the `dataset/` directory.

The dataset for the project follows this structure:

```text
dataset/
  pretrain/
    train_data/
    val_data/
    test_data/
  forecasting/
    train.json
    val.json
    test.json
  retrieval/
    train.parquet
    test.parquet
```

## 3) Project Organization

```text
├── pretrain.py                  # Stage 1
├── context_align.py             # Stage 2
├── forecast_finetune.py         # Optional for task-specific finetuning
├── demo.ipynb                   # Embedding + retrieval demo
├── configs/
│   ├── pretrain.yaml
│   ├── align.yaml
│   └── finetune.yaml
└── src/
    ├── data/                    # Dataset + dataloader
    ├── models/                  # TS encoder / multimodal encoder / retriever
    ├── tasks/                   # Training loops
    └── utils/                   # Config / metrics / helpers
```

## 4) Training

TRACE uses a two-stage training pipline. Stage 3 serves as an optional stage for task-specific finetuning.
- Stage 1: time-series pretraining
- Stage 2: time-series/text context alignment (embedding + retrieval)
- Stage 3 (optional): forecasting finetuning (with or without RAG)

![TimeSeriesRAG Pipeline](misc/pipeline.png)


### Stage 1: Pretrain

```bash
python pretrain.py \
  --config configs/pretrain.yaml
```

After pretraining, record the run name.

Important:

- This run name is the key link to Step 2.
- `context_align.py` uses `--pretraining_run_name` to locate and override model settings from `results/wandb_configs/<PRETRAIN_RUN_NAME>.yaml`.
- Pretraining checkpoints are expected under `results/model_checkpoints/<PRETRAIN_RUN_NAME>/`.

### Stage 2: Context Align

```bash
python context_align.py \
  --config configs/align.yaml \
  --pretraining_run_name "<PRETRAIN_RUN_NAME>" \
  --cross_attend
```

## 5) Optional Forecast Finetune

Use the same `<PRETRAIN_RUN_NAME>` from Stage 1.

### 5.1 w/o RAG (TS-only)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master-port=<MASTER_PORT_FT_WO_RAG> \
  forecast_finetune.py \
  --config configs/finetune.yaml \
  --pretraining_run_name "<PRETRAIN_RUN_NAME>" \
  --ts_only
```

### 5.2 w/ RAG (TS + Text)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master-port=<MASTER_PORT_FT_W_RAG> \
  forecast_finetune.py \
  --config configs/finetune.yaml \
  --pretraining_run_name "<PRETRAIN_RUN_NAME>" \
  --top_k 1
```

## Retrieval Demo
Refer to  `demo.ipynb` for generating embedding bank and cross-modal retrieval.

![TimeSeriesRAG Retrieval Demo](misc/retrieval.png)

## Citation
If you find this work useful, please consider citing our paper:
```bibtex
@article{chen2025trace,
  title={Trace: Grounding time series in context for multimodal embedding and retrieval},
  author={Chen, Jialin and Zhao, Ziyu and Nurbek, Gaukhar and Feng, Aosong and Maatouk, Ali and Tassiulas, Leandros and Gao, Yifeng and Ying, Rex},
  journal={arXiv preprint arXiv:2506.09114},
  year={2025}
}
```
