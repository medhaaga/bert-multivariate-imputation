import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append('../')
sys.path.append('../../')

from src.os_utils import (get_data_path)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def combine_raw_data():
    data_dir = get_data_path()
    files = [file for file in os.listdir(os.path.join(data_dir, 'raw_data')) if file.endswith('.csv')]
    df = []
    for file in files:
        tmp = pd.read_csv(os.path.join(data_dir, 'raw_data', file), index_col='No')
        df.append(tmp)
    df = pd.concat(df, ignore_index=True)
    df.to_csv(os.path.join(data_dir, 'data.csv'))


class MLMDataset:
    def __init__(self, temporal, static):
        self.temporal = torch.tensor(temporal, dtype=torch.float32)
        self.static = torch.tensor(static, dtype=torch.float32)

    def __len__(self,):
        return len(self.temporal)

    def __getitem__(self,idx):
        return self.temporal[idx], self.static[idx]

def time_series_linear_interpolation(data):
    """
    Perform linear interpolation on a multivariate time series data.

    Parameters:
    - data: A numpy array of shape (N, T, P)

    Returns:
    - data_imputed: A numpy array of shape (N, T, P) with missing values filled
    """

    N, T, P = data.shape
    data_imputed = np.copy(data)

    # Iterate over each series and observation
    for p in range(P):
        for n in range(N):

            series = data[n, :, p]
            not_nan_indices = np.where(~np.isnan(series))[0]
            nan_indices = np.where(np.isnan(series))[0]
            
            if len(not_nan_indices) > 1:  # Need at least two points to interpolate
                data_imputed[n, nan_indices, p] = np.interp(
                    nan_indices, not_nan_indices, series[not_nan_indices]
                )
            elif len(not_nan_indices) == 1:
                # If only one non-nan point, fill with that value
                data_imputed[n, nan_indices, p] = series[not_nan_indices[0]]
            else:
                # If all are nan, leave as nan or fill with a default value (e.g., zero)
                data_imputed[n, nan_indices, p] = 0.0  # or use np.nanmean(series) if appropriate

    return data_imputed

def static_linear_interpolation(data):
    
    """Perform linear interpolation on a multivariate tabular data.

    Parameters:
    - data: A numpy array of shape (N, P)

    Returns:
    - data_imputed: A numpy array of shape (N, P) with missing values filled
    """
    N, P = data.shape
    data_imputed = np.copy(data)

    # Iterate over each series and observation
    for p in range(P):
            series = data[:, p]
            not_nan_indices = np.where(~np.isnan(series))[0]
            nan_indices = np.where(np.isnan(series))[0]

            if len(not_nan_indices) > 1:  # Need at least two points to interpolate
                data_imputed[nan_indices, p] = np.interp(
                    nan_indices, not_nan_indices, series[not_nan_indices]
                )
            elif len(not_nan_indices) == 1:
                # If only one non-nan point, fill with that value
                data_imputed[nan_indices, p] = series[not_nan_indices[0]]
            else:
                # If all are nan, leave as nan or fill with a default value (e.g., zero)
                data_imputed[nan_indices, p] = 0.0  # or use np.nanmean(series) if appropriate

    return data_imputed

def create_dataset(test_ratio=0.15, val_ratio=0.15):

    """Create dataset using hourly observations for each day.
    
    Arguments:
        test_ratio (float) = 0.15: Proportion of the dataset to include in the test split.
        val_ratio (float) = 0.15: Proportion of the dataset to include in the validation split (relative to the training+validation data).

    Returns:
        dict: A dictionary containing PyTorch tensors for training, validation, and test sets:
            - 'X_train': Training temporal data.
            - 'y_train': Training static data.
            - 'X_val': Validation temporal data.
            - 'y_val': Validation static data.
            - 'X_test': Test temporal data.
            - 'y_test': Test static data.

    """

    # if pre-saved data already exists...
    if os.path.exists(os.path.join(get_data_path(), 'data.pth')):
        data = torch.load(os.path.join(get_data_path(), 'data.pth'))

    else:
        df = pd.read_csv(os.path.join(get_data_path(), 'data.csv'))
        df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['station'] = df['station'].astype('category')
        label_encoder = LabelEncoder()
        df['encoded_station'] = label_encoder.fit_transform(df['station'])
        groups = df.groupby('encoded_station')

        temporal_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        temporal_data, static_data = [], []

        for grp_name, grp_df in groups:
            start_idx = 0
            while (start_idx + 168 <= len(grp_df)):
                temp = grp_df.iloc[start_idx:(start_idx+168)]
                temporal_temp = np.array(temp[temporal_cols].values, dtype=np.float32)
                
                start_idx += 168
     
                temporal_data.append(temporal_temp)

                static_row = np.array([min(temp.TEMP), max(temp.TEMP), grp_name, np.mean(temp.PRES)], dtype=np.float32)
                static_data.append(static_row)
                
        temporal_data = np.array(temporal_data)
        static_data = np.array(static_data)

        # standardize data
        std = np.nanstd(temporal_data, axis=1, keepdims=True) + 1e-8
        temporal_data = temporal_data/std

        # remove extreme outliers
        mask = np.all(temporal_data <= 1e4, axis=(1, 2))
        temporal_data = temporal_data[mask]
        static_data = static_data[mask]

        # there must already by many missing values in the train and val set. Pad them to ignore for loss calculation
        temporal_data =  time_series_linear_interpolation(temporal_data) #config['pad_token_id']
        static_data = static_linear_interpolation(static_data) #config['pad_token_id']

        # First split: Train+Val and Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(temporal_data, static_data, test_size=test_ratio, random_state=42)
        val_size = val_ratio / (1 - test_ratio)  # Adjusted validation size relative to the remaining data
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)

  
        data = {'X_train': X_train, 'y_train': y_train,
                    'X_val': X_val, 'y_val': y_val,
                    'X_test': X_test, 'y_test': y_test}

        torch.save(data, os.path.join(get_data_path(), 'data.pth'))

    
    train_dataset = MLMDataset(data['X_train'], data['y_train'])
    val_dataset = MLMDataset(data['X_val'], data['y_val'])
    test_dataset = MLMDataset(data['X_test'], data['y_test'])

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    combine_raw_data()