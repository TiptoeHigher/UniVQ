import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')

class Dataset_Pretrain_test(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.min_pred_len = 96
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)

        # split
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

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values


        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.num_channels = self.data_x.shape[-1]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)


    def __getitem__(self, index):
        total_len = len(self.data_x) - self.seq_len - self.min_pred_len + 1
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end_96 = r_begin + self.label_len + 96
        r_end_192 = r_begin + self.label_len + 192
        r_end_336 = r_begin + self.label_len + 336
        r_end_720 = r_begin + self.label_len + 720

        seq_x = self.data_x[s_begin:s_end]

        seq_y_96 = self.data_y[r_begin:r_end_96]

        if r_end_192 < total_len:
            seq_y_192 = self.data_y[r_begin:r_end_192]
        else:
            seq_y_192 = np.zeros((self.label_len + 192, self.num_channels))

        if r_end_336 < total_len:
            seq_y_336 = self.data_y[r_begin:r_end_336]
        else:
            seq_y_336 = np.zeros((self.label_len + 336, self.num_channels))

        if r_end_720 < total_len:
            seq_y_720 = self.data_y[r_begin:r_end_720]
        else:
            seq_y_720 = np.zeros((self.label_len + 720, self.num_channels))


        return seq_x, seq_y_96, seq_y_192, seq_y_336, seq_y_720

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.min_pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)