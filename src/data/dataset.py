import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.common import PATHS, TASKS
from src.data.load_data import load_timeseries_from_json, load_npy_timeseries, load_forecasting_from_json, load_retrieval_from_parquet
from src.utils.data import (
    interpolate_timeseries,
    upsample_timeseries,
)
from .base import TaskDataset, TimeseriesData
import torch
warnings.filterwarnings("ignore")


class PretrainingDataset(TaskDataset):
    def __init__(
        self,
        seq_len_channel: int = 180,
        root_path: str = PATHS.DATA_DIR + "pretrain/",
        data_split: str = "train",
        scale: bool = True,
        task_name: str = TASKS.PRETRAINING,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        upsampling_pad_direction="backward",
        upsampling_type="pad",
        downsampling_type="interpolate",
        pad_mode="constant",
        pad_constant_values=0,
        return_meta_data=False,
        **kwargs,
    ):
        super(PretrainingDataset, self).__init__()
        self.seq_len_channel = seq_len_channel
        self.root_path = root_path

        self.data_split = data_split
        self.scale = scale
        self.task_name = task_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.upsampling_pad_direction = upsampling_pad_direction
        self.upsampling_type = upsampling_type
        self.downsampling_type = downsampling_type
        self.pad_mode = pad_mode
        self.pad_constant_values = pad_constant_values
        self.return_meta_data = return_meta_data
        # Input checking
        self._check_inputs()

        # Read data
        self._read_data()

    def _check_inputs(self):
        # Input checking
        assert self.data_split in [
            "train",
            "test",
            "val",
        ], "data_split must be one of 'train', 'test' or 'val'"


    def _transform_labels(self, train_labels: np.ndarray, test_labels: np.ndarray):
        # Move the labels to {0, ..., L-1}
        labels = np.unique(train_labels)
        transform = {}
        for i, l in enumerate(labels):
            transform[l] = i

        train_labels = np.vectorize(transform.get)(train_labels)
        test_labels = np.vectorize(transform.get)(test_labels)

        return train_labels, test_labels

    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()

        # output = load_timeseries_from_json(self.data_split, self.root_path, return_meta_data=self.return_meta_data)
        output = load_npy_timeseries(self.data_split, self.root_path, return_meta_data=self.return_meta_data)
        
        if self.return_meta_data:
            self.data, self.labels, self.meta_data = output
        else:
            self.data, self.labels = output 
        
        # meta_data: list of dicts
        # Check if time-series have equal lengths. If not, left pad with zeros
        # self._check_if_equal_length()

        # Check and remove NaNs
        self._check_and_remove_nans()
        self.n_timeseries = len(self.data)
        if self.scale:
            for i, ts in enumerate(self.data):
                ts = ts.T  # Now shape is [L, C]
                ts_scaled = self.scaler.fit_transform(ts)  
                self.data[i] = ts_scaled.T
        # self.data: list of [C, L], L varies across time
        # self.data = self.data.T
        # self.input_mask = self.input_mask.T

    def __getitem__(self, index):
        assert index < self.__len__()

        timeseries = self.data[index] # [C, L]
        timeseries_len = timeseries.shape[1]
        labels = self.labels[index,].astype(int)
        
        ## padding to the same length
        if timeseries_len <= self.seq_len_channel:
            timeseries, input_mask = upsample_timeseries(
                timeseries,
                self.seq_len_channel,
                direction=self.upsampling_pad_direction,
                sampling_type=self.upsampling_type,
                mode=self.pad_mode,
            )

        meta_data = self.meta_data[index] if self.return_meta_data else None
        
        return TimeseriesData(
            timeseries=timeseries,  # [C, L]
            labels=labels,
            input_mask=input_mask,  # [C,L]
            metadata=meta_data
        )

    def __len__(self):
        return self.n_timeseries

    def _check_and_remove_nans(self):
        for i, ts in enumerate(self.data):
            if np.isnan(ts).any():
                ts = interpolate_timeseries(
                    timeseries=ts, interp_length=ts.shape[-1]
                )
                ts = np.nan_to_num(ts)
                self.data[i] = ts

    def _check_if_equal_length(self):
        if isinstance(self.data, list):
            n_timeseries = len(self.data)
            self.n_channels = self.data[0].shape[0]
            # Assume all time-series have the same number of channels
            # Then we have time-series of unequal lengths
            max_len = max([ts.shape[-1] for ts in self.data])
            for i, ts in enumerate(self.data):
                self.data[i] = interpolate_timeseries(
                    timeseries=ts, interp_length=max_len
                )
            self.data = np.asarray(self.data)
            logging.info(
                f"Time-series have unequal lengths. Reshaping to {self.data.shape}"
            )

    def plot(self, idx, channel=0):
        timeseries_data = self.__getitem__(idx)
        label = timeseries_data.labels
        timeseries = timeseries_data.timeseries[0, channel, :]

        plt.title(f"idx={idx}, label={label}", fontsize=18)
        plt.plot(
            np.arange(self.seq_len_channel),
            timeseries,
            label="Time-series",
            c="darkblue",
        )
        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Value", fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=18)
        plt.show()


class ForecastingDataset(TaskDataset):
    def __init__(
        self,
        seq_len_channel: int = 180,
        forecast_horizon: int = 7,
        root_path: str = PATHS.DATA_DIR + "forecasting/",
        data_split: str = "train",
        scale: bool = True,
        task_name: str = TASKS.FORECASTING,
        upsampling_pad_direction="backward",
        upsampling_type="pad",
        downsampling_type="interpolate",
        pad_mode="constant",
        pad_constant_values=0,
        return_meta_data=False,
        **kwargs,
    ):
        super(ForecastingDataset, self).__init__()
        self.seq_len_channel = seq_len_channel
        self.forecast_len = forecast_horizon
        self.root_path = root_path
        self.data_split = data_split
        self.scale = scale
        self.task_name = task_name
        self.upsampling_pad_direction = upsampling_pad_direction
        self.upsampling_type = upsampling_type
        self.downsampling_type = downsampling_type
        self.pad_mode = pad_mode
        self.pad_constant_values = pad_constant_values
        self.return_meta_data = return_meta_data
        # Input checking
        self._check_inputs()

        # Read data
        self._read_data()

    def _check_inputs(self):
        # Input checking
        assert self.data_split in [
            "train",
            "test",
            "val",
        ], "data_split must be one of 'train', 'test' or 'val'"



    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()

        raw_data= load_forecasting_from_json(self.data_split, self.root_path)
        data = []
        forecast_data = []
        # Check and remove NaNs
        raw_data = self._check_and_remove_nans(raw_data)
        self.n_timeseries = len(raw_data)
        if self.scale:
            for i, ts in enumerate(raw_data):
                ts_scale = self.scaler.fit_transform(ts.T)
                ts_scale = ts_scale.T  #[C, L]
                data.append(ts_scale[:, :-self.forecast_len])
                forecast_data.append(ts_scale[:, -self.forecast_len:])
        self.data = data
        self.forecast_data = forecast_data

    def __getitem__(self, index):
        assert index < self.__len__()

        timeseries = self.data[index] # [C, L]
        forecast = self.forecast_data[index] # [C, H]
        assert forecast.shape[-1] == self.forecast_len  
        timeseries_len = timeseries.shape[1]
        
        ## padding to the same length
        if timeseries_len <= self.seq_len_channel:
            timeseries, input_mask = upsample_timeseries(
                timeseries,
                self.seq_len_channel,
                direction=self.upsampling_pad_direction,
                sampling_type=self.upsampling_type,
                mode=self.pad_mode,
            )

        return TimeseriesData(
            timeseries=timeseries,  # [C, L]
            forecast=forecast,  # [C, H]
            input_mask=input_mask,  # [C,L]
        )

    def __len__(self):
        return self.n_timeseries

    def _check_and_remove_nans(self, data):
        for i, ts in enumerate(data):
            if np.isnan(ts).any():
                ts = interpolate_timeseries(
                    timeseries=ts, interp_length=ts.shape[-1]
                )
                ts = np.nan_to_num(ts)
                data[i] = ts
        return data



class ClassificationDataset(TaskDataset):
    def __init__(self, 
        seq_len_channel: int = 180,
        root_path: str = PATHS.DATA_DIR + "classification/",
        data_split: str = "train",
        scale: bool = True,
        upsampling_pad_direction="backward",
        upsampling_type="pad",
        pad_mode="constant"):
        super(ClassificationDataset, self).__init__()
        self.seq_len_channel = seq_len_channel
        self.root_path = root_path
        self.data_split = data_split
        self.scale = scale
        self.upsampling_pad_direction = upsampling_pad_direction
        self.upsampling_type = upsampling_type
        self.pad_mode = pad_mode
        self._read_data()
    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()
        self.data, self.labels = load_npy_timeseries(self.data_split, self.root_path)
        self._check_and_remove_nans()
        self.n_timeseries = len(self.data)
        if self.scale:
            for i, ts in enumerate(self.data):
                ts = ts.T  # Now shape is [L, C]
                ts_scaled = self.scaler.fit_transform(ts)  
                self.data[i] = ts_scaled.T
        
        
    def __getitem__(self, index):
        timeseries = self.data[index] # [C, L]
        timeseries_len = timeseries.shape[1]
        labels = self.labels[index,].astype(int)
        
        ## padding to the same length
        if timeseries_len <= self.seq_len_channel:
            timeseries, input_mask = upsample_timeseries(
                timeseries,
                self.seq_len_channel,
                direction=self.upsampling_pad_direction,
                sampling_type=self.upsampling_type,
                mode=self.pad_mode,
            )
        
        return TimeseriesData(
            timeseries=timeseries,  # [C, L]
            labels=labels,
            # input_mask=input_mask,  # [C,L]
        )

    def _check_and_remove_nans(self):
        for i, ts in enumerate(self.data):
            if np.isnan(ts).any():
                ts = interpolate_timeseries(
                    timeseries=ts, interp_length=ts.shape[-1]
                )
                ts = np.nan_to_num(ts)
                self.data[i] = ts
                
    def __len__(self):
        return self.n_timeseries
    
    
class RetrievalDataset(TaskDataset):
    def __init__(self, 
        seq_len_channel: int = 180,
        root_path: str = PATHS.DATA_DIR + "retrieval/",
        data_split: str = "train",
        scale: bool = True,
        text_encoder_name: str = "bert-base-uncased",
        upsampling_pad_direction="backward",
        upsampling_type="pad",
        pad_mode="constant"):
        super(RetrievalDataset, self).__init__()
        self.seq_len_channel = seq_len_channel
        self.root_path = root_path
        self.data_split = data_split
        self.scale = scale
        self.upsampling_pad_direction = upsampling_pad_direction
        self.upsampling_type = upsampling_type
        self.pad_mode = pad_mode
        self.text_encoder_name = text_encoder_name
        self._read_data()

    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()
        if self.data_split == "train":
            self.data, self.descriptions_emb, self.channel_descriptions_emb, self.events_emb, self.labels= load_retrieval_from_parquet(self.data_split, self.root_path, self.text_encoder_name)
        else:
            self.data, self.descriptions_emb, self.channel_descriptions_emb, self.events_emb, self.labels, self.descriptions, self.channel_descriptions, self.events= load_retrieval_from_parquet(self.data_split, self.root_path, self.text_encoder_name)
        self._check_and_remove_nans()
        self.n_timeseries = len(self.data)
        if self.scale:
            for i, ts in enumerate(self.data):
                ts = ts.T  # Now shape is [L, C]
                ts_scaled = self.scaler.fit_transform(ts)  
                self.data[i] = ts_scaled.T
        
        
    def __getitem__(self, index):
        timeseries = self.data[index] # [C, L]
        timeseries_len = timeseries.shape[1]
        labels = self.labels[index,].astype(int)
        ## padding to the same length
        if timeseries_len <= self.seq_len_channel:
            timeseries, input_mask = upsample_timeseries(
                timeseries,
                self.seq_len_channel,
                direction=self.upsampling_pad_direction,
                sampling_type=self.upsampling_type,
                mode=self.pad_mode,
            )
        
        if self.data_split == "train":
            return TimeseriesData(
                timeseries=timeseries,  # [C, L]
                labels=labels,
                description_emb=self.descriptions_emb[index],
                channel_description_emb=self.channel_descriptions_emb[index], #[C, d]
                event_emb=self.events_emb[index],
                input_mask=input_mask,  # [C,L]
            )
        else:
            return TimeseriesData(
                timeseries=timeseries,  # [C, L]
                labels=labels,
                description_emb=self.descriptions_emb[index],
                channel_description_emb=self.channel_descriptions_emb[index], #[C, d]
                event_emb=self.events_emb[index],
                descriptions=self.descriptions[index],
                # channel_descriptions=self.channel_descriptions[index],
                events=self.events[index],
                input_mask=input_mask,  # [C,L]
            )

    def _check_and_remove_nans(self):
        for i, ts in enumerate(self.data):
            if np.isnan(ts).any():
                ts = interpolate_timeseries(
                    timeseries=ts, interp_length=ts.shape[-1]
                )
                ts = np.nan_to_num(ts)
                self.data[i] = ts
                
    def __len__(self):
        return self.n_timeseries





class MMDataset(TaskDataset):
    def __init__(
        self,
        seq_len_channel: int = 32,
        forecast_len: int = 12,
        root_path: str = PATHS.DATA_DIR,
        data_name: str = "env",
        data_split: str = "train",
        scale: bool = True,
        task_name: str = TASKS.PRETRAINING,
        **kwargs,
    ):
        super(MMDataset, self).__init__()
        self.seq_len = seq_len_channel
        self.label_len = forecast_len
        self.pred_len = forecast_len
        assert data_split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[data_split]
        self.text_name='Final_Search_6'
        self.features = "S"
        self.target = "OT"
        self.scale = scale
        assert data_name in ["env", "health", "energy"]
        self.root_path = root_path + f"{data_name}/"
        self.data_path = f"{data_name}.csv"
        self.__read_data__()
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        
        df_raw = df_raw[['date'] + cols + [self.target]+['prior_history_avg']+['start_date']+['end_date']+[self.text_name]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_data_prior = df_raw[['prior_history_avg']]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]  
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            data_prior = self.scaler.transform(df_data_prior.values[:,-1].reshape(-1, 1))
        else:
            data = df_data.values
            data_prior = df_data_prior.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_prior = data_prior[border1:border2]

        self.date=df_raw[['date']][border1:border2].values
        self.start_date=df_raw[['start_date']][border1:border2].values
        self.end_date=df_raw[['end_date']][border1:border2].values
        self.text=df_raw[[self.text_name]][border1:border2].values
        
    def get_prior_y(self, indices):
        # If indices is a single integer index
        if isinstance(indices, (int, np.integer)):
            s_begin = indices % self.tot_len
            s_end = s_begin + self.seq_len
            r_begin = s_end
            r_end = r_begin + self.pred_len
            prior_y = self.data_prior[r_begin:r_end]
            return prior_y
        
        # If indices is a tensor or array
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
            
        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends
        r_ends = r_begins + self.pred_len
        prior_y = np.array([self.data_prior[r_beg:r_end] for r_beg, r_end in zip(r_begins, r_ends)])
        return prior_y
    
    ## TODO: no need for pretraining => revise for retrieval task
    # def get_text(self, indices):
    #     if isinstance(indices, torch.Tensor):
    #         indices = indices.numpy()

    #     s_begins = indices % self.tot_len
    #     s_ends = s_begins + self.seq_len
    #     print(s_ends)
    #     text=np.array([self.text[s_end] for s_end in s_ends])[:,0]
    #     return text
    
    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1].reshape(1, -1)

        # Fix for r_begin < 0
        if r_begin >= 0:
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1].reshape(1, -1)
        else:
            valid_part = self.data_y[0:r_end, feat_id:feat_id + 1]  # shape: [r_end, 1]
            pad_len = abs(r_begin)
            pad_part = np.zeros((pad_len, 1))  # shape: [pad_len, 1]
            seq_y = np.concatenate([pad_part, valid_part], axis=0).reshape(1, -1)

        input_mask = np.ones((1, self.seq_len))
        prior_y = self.get_prior_y(index).reshape(1, -1)

        return TimeseriesData(
            timeseries=seq_x,
            input_mask=input_mask,
            forecast=seq_y,
            prior_y=prior_y
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)