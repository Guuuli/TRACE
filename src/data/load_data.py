

import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io
import torch
from src.common import EVENT_MAP
from tqdm import tqdm
keys_to_save = ['temperature', 'precipitation', 'relative_humidity', 'visibility', 'wind_u', 'wind_v', 'sky_code']

def load_timeseries_from_json(split: str, dir_path: str, return_meta_data=False):
    file_path = os.path.join(dir_path, f'{split}.json')
    ts_data = []
    labels = []
    meta_data = []
    with open(file_path, 'r') as f:
        data = json.load(f)
    for i, (k, v) in enumerate(list(data.items())):
        ts_sample = []
        for key in keys_to_save:
            if len(v[key]) >= 100:
                ts_sample.append(v[key])
            
        if len(ts_sample) > 0:
            ts_sample = np.array(ts_sample)
            ts_data.append(ts_sample)
            labels.append(v['event_type'])
            meta_data.append({"id": k, "station_id": v["station_id"], "mode": v['mode'], "location": v['location']})
                
    labels = np.array(labels).reshape(-1, 1)
    if return_meta_data:
        return ts_data, labels, meta_data
    else:
        return ts_data, labels


def load_npy_timeseries(split: str, dir_path: str, return_meta_data=False):
    file_path = os.path.join(dir_path, f'{split}_data')
    ts_data = []
    # Load all numbered npy files (timeseries data)
    npy_files = [f for f in os.listdir(file_path) if f.endswith('.npy') and f != 'labels.npy']
    npy_files.sort()  # Ensure consistent ordering
    
    for npy_file in npy_files:
        ts = np.load(os.path.join(file_path, npy_file))
        ts_data.append(ts)
        
    # Load labels
    labels = np.load(os.path.join(file_path, 'labels.npy'))
    
    return ts_data, labels

def load_forecasting_from_json(split: str, dir_path: str):
    file_path = os.path.join(dir_path, f'{split}.json')
    ts_data = []
    with open(file_path, 'r') as f:
        data = json.load(f)
    for i, (k, v) in enumerate(list(data.items())[:]):
        ts_sample = []
        for key in keys_to_save:
            if len(v[key]) >= 100:
                ts_sample.append(v[key])
            
        if len(ts_sample) > 0:
            ts_sample = np.array(ts_sample)
            ts_data.append(ts_sample)

    return ts_data


def generate_dsp(description):
    keys_to_save = ['temperature', 'precipitation', 'relative_humidity', 'visibility', 'wind_u', 'wind_v', 'sky_code']
    date = description["DATE"]
    location = description["location"]
    labels = description["labels"]
    prompt = f"Weather time series location: {location} Time range: {date} The weather is {labels}. {description[keys_to_save[0]]} \n {description[keys_to_save[1]]} \n {description[keys_to_save[2]]} \n {description[keys_to_save[3]]} \n {description[keys_to_save[4]]} \n {description[keys_to_save[5]]} \n {description[keys_to_save[6]]}"
    return prompt

def generate_channel_description(description):
    keys_to_save = ['temperature', 'precipitation', 'relative_humidity', 'visibility', 'wind_u', 'wind_v', 'sky_code']
    channel_description = [description[key] for key in keys_to_save]
    return channel_description
    
    
    
def generate_er(event):
    event_idx = int(event["event_type"])
    event_type = list(EVENT_MAP.keys())[event_idx]
    event_description = event["narrative"]
    prompt = f"The weather event is {event_type}. {event_description}"
    return prompt

def load_retrieval_from_parquet(split: str, file_path: str, text_encoder_name: str, device="cuda:0"):
    file_path_pq = os.path.join(file_path+split, f'{split}.parquet')
    df = pd.read_parquet(file_path_pq)
    timeseries = []
    descriptions = []
    channel_descriptions = []
    events = []
    labels = []
    for ts_bytes in df["timeseries"]:
        ts = np.load(io.BytesIO(ts_bytes))
        timeseries.append(ts)
    for description in df["description"]:
        context = generate_dsp(description)
        channel_description = generate_channel_description(description)
        descriptions.append(context)
        channel_descriptions.extend(channel_description)
    for event in df["events"]:
        if event is not None:
            er = generate_er(event)
            events.append(er)
            labels.append(int(event["event_type"]))
        else:
            events.append("No severe weather event.")
            labels.append(-100)
    labels = np.array(labels).reshape(-1, 1)
    assert len(descriptions)*len(keys_to_save) == len(channel_descriptions)
    
    encoder_short_name = text_encoder_name.split("/")[-1]
    
    channel_description_emb_path = os.path.join(file_path+split, f'channel_description_emb_{encoder_short_name}.pt')
    if os.path.exists(channel_description_emb_path):
        channel_description_emb = torch.load(channel_description_emb_path,map_location="cpu")
    else:
        from sentence_transformers import SentenceTransformer
        print(f"Generating channel description embeddings with {text_encoder_name}...")
        model = SentenceTransformer(text_encoder_name, trust_remote_code=True)
        channel_description_emb = model.encode(
            channel_descriptions, 
            batch_size=64,         
            show_progress_bar=True,
            convert_to_tensor=True   
        )
        torch.save(channel_description_emb, channel_description_emb_path)
    
    description_emb_path = os.path.join(file_path+split, f'description_emb_{encoder_short_name}.pt')
    if os.path.exists(description_emb_path):
        description_emb = torch.load(description_emb_path,map_location="cpu")
    else:
        from sentence_transformers import SentenceTransformer
        print(f"Generating description embeddings with {text_encoder_name}...")
        model = SentenceTransformer(text_encoder_name, trust_remote_code=True)
        description_emb = model.encode(
            descriptions, 
            batch_size=64,         
            show_progress_bar=True,
            convert_to_tensor=True 
        )
        torch.save(description_emb, description_emb_path)
        
    
    event_emb_path = os.path.join(file_path+split, f'event_emb_{encoder_short_name}.pt')
    if os.path.exists(event_emb_path):
        event_emb = torch.load(event_emb_path,map_location="cpu")
    else:
        from sentence_transformers import SentenceTransformer
        print(f"Generating event embeddings with {text_encoder_name}...")
        model = SentenceTransformer(text_encoder_name, trust_remote_code=True)
        event_emb = model.encode(
            events, 
            batch_size=64,         
            show_progress_bar=True,
            convert_to_tensor=True   
        )
        torch.save(event_emb, event_emb_path)
    emb_dim = event_emb.shape[1]
    channel_description_emb = channel_description_emb.reshape(-1, len(keys_to_save), emb_dim)
    assert channel_description_emb.shape[0] == description_emb.shape[0] == event_emb.shape[0]
    if split == "train":
        return timeseries, description_emb, channel_description_emb, event_emb, labels
    else:
        return timeseries, description_emb, channel_description_emb, event_emb, labels, descriptions, channel_descriptions, events



