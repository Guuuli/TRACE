import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .base import TimeseriesData
from .dataset import PretrainingDataset, ForecastingDataset, ClassificationDataset, RetrievalDataset, MMDataset
from torch.utils.data.distributed import DistributedSampler


def _collate_fn_basic(examples):
    examples = list(filter(lambda x: x is not None, examples))
    timeseries = [torch.from_numpy(example.timeseries) for example in examples] # [C, L]
    input_masks = [torch.from_numpy(example.input_mask) for example in examples] # [C, L]
    timeseries = torch.stack(timeseries)  # [B, C, L]
    input_masks = torch.stack(input_masks)  # [B, C, L]
    labels = [example.labels for example in examples]
    labels = np.asarray(labels)  # [B]
    return TimeseriesData(timeseries=timeseries, input_mask=input_masks, labels=labels)

def _collate_fn_forecasting(examples):
    examples = list(filter(lambda x: x is not None, examples))
    timeseries = [torch.from_numpy(example.timeseries) for example in examples] # [C, L]
    input_masks = [torch.from_numpy(example.input_mask) for example in examples] # [C, L]
    timeseries = torch.stack(timeseries)  # [B, C, L]
    input_masks = torch.stack(input_masks)  # [B, C, L]
    forecast = [torch.from_numpy(example.forecast) for example in examples] # [C, H]
    forecast = torch.stack(forecast)  # [B, C, H]
    
    if examples[0].prior_y is not None:
        prior_y = [torch.from_numpy(example.prior_y) for example in examples] # [C, H]
        prior_y = torch.stack(prior_y)  # [B, C, H]
        return TimeseriesData(timeseries=timeseries, input_mask=input_masks, forecast=forecast, prior_y=prior_y)
    else:
        return TimeseriesData(timeseries=timeseries, input_mask=input_masks, forecast=forecast)

def _collate_fn_classification(examples):
    examples = list(filter(lambda x: x is not None, examples))
    timeseries = [torch.from_numpy(example.timeseries) for example in examples] # [C, L]
    timeseries = torch.stack(timeseries)  # [B, C, L]
    labels = [example.labels for example in examples]
    labels = np.asarray(labels)  # [B]
    return TimeseriesData(timeseries=timeseries, labels=labels)


def _collate_fn_retrieval(examples):
    examples = list(filter(lambda x: x is not None, examples))
    timeseries = [torch.from_numpy(example.timeseries) for example in examples] # [C, L]
    timeseries = torch.stack(timeseries)  # [B, C, L]
    
    input_masks = [torch.from_numpy(example.input_mask) for example in examples] # [C, L]
    input_masks = torch.stack(input_masks)  # [B, C, L]
    
    labels = [example.labels for example in examples]
    labels = np.asarray(labels)  # [B]
    
    channel_description_emb = [example.channel_description_emb for example in examples]
    channel_description_emb = torch.stack(channel_description_emb)  # [B, C, d]
    
    description_emb = [example.description_emb for example in examples]
    description_emb = torch.stack(description_emb)  # [B, d]
    
    event_emb = [example.event_emb for example in examples]
    event_emb = torch.stack(event_emb)  # [B, d]
    
    if hasattr(examples[0], 'descriptions'): #in test set
        descriptions = [example.descriptions for example in examples]
        events = [example.events for example in examples]
        return TimeseriesData(timeseries=timeseries,
                              input_mask=input_masks, 
                              labels=labels, 
                              channel_description_emb=channel_description_emb, 
                              description_emb=description_emb, 
                              event_emb=event_emb, 
                              descriptions=descriptions, 
                              events=events)
    else:
        return TimeseriesData(timeseries=timeseries,
                              input_mask=input_masks, 
                              labels=labels, 
                              channel_description_emb=channel_description_emb, 
                              description_emb=description_emb, 
                              event_emb=event_emb)

def get_dataloader(args):
    if hasattr(args, "data_name") and args.data_name in ["health", "env", "energy"]:
        return get_mmd_dataloader(args)
    else:
        if args.task_name == "pretraining":
            dataset = PretrainingDataset(
                seq_len_channel=args.seq_len_channel,
                data_split=args.data_split,
                scale=args.scale,
                upsampling_pad_direction=args.upsampling_pad_direction,
                upsampling_type=args.upsampling_type,
                downsampling_type=args.downsampling_type,
                pad_mode=args.pad_mode,
            )
        elif args.task_name == "forecasting":
            dataset = ForecastingDataset(
                seq_len_channel=args.seq_len_channel,
                forecast_horizon=args.forecast_horizon,
                data_split=args.data_split,
                scale=args.scale,
                upsampling_pad_direction=args.upsampling_pad_direction,
                upsampling_type=args.upsampling_type,
                downsampling_type=args.downsampling_type,
                pad_mode=args.pad_mode,
            )
        elif args.task_name == "classification":
            dataset = ClassificationDataset(
                seq_len_channel=args.seq_len_channel,
                data_split=args.data_split,
                scale=args.scale,
            )
        elif args.task_name == "retrieval":
            dataset = RetrievalDataset(
                seq_len_channel=args.seq_len_channel,
                data_split=args.data_split,
                scale=args.scale,
                text_encoder_name=args.text_encoder_name,
            )
        else:
            raise ValueError(f"Invalid task name: {args.task_name}")
        
        collate_fn_map = {
            "pretraining": _collate_fn_basic,
            "forecasting": _collate_fn_forecasting,
            "classification": _collate_fn_classification,
            "retrieval": _collate_fn_retrieval,
        }
        if getattr(args, "distributed", False):
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                collate_fn=collate_fn_map[args.task_name],
                sampler=DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=args.shuffle),
            )
        else:
            shuffle = True if args.data_split == "train" else False
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=shuffle,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                collate_fn=collate_fn_map[args.task_name],
            )
        return dataloader



def get_mmd_dataloader(args):
    dataset = MMDataset(
        seq_len_channel=args.seq_len_channel,
        forecast_len=args.forecast_horizon,
        data_name=args.data_name,
        data_split=args.data_split,
        scale=args.scale)
    
    if args.world_size > 1:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=_collate_fn_forecasting,
            sampler=DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=args.shuffle),
            )
    else:
        shuffle = True if args.data_split == "train" else False
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=_collate_fn_forecasting
        )
    return dataloader