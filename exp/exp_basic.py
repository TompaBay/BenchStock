import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

# from data_provider.data_factory import data_provider
from exp.exp import Exp
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, r2

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

import os
import time
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')


class TrainDataset(Dataset):
    # load the dataset
    def __init__(self, x, y):#, context = 0):
        self.X = x
        self.Y = y

    # get number of items/rows in dataset
    def __len__(self):
        return len(self.Y)

    # get row item at some index
    def __getitem__(self, index):
        x = torch.FloatTensor(self.X[index].reshape(-1))
        y = torch.FloatTensor(self.Y[index])
        return x, y


class Exp_Basic(Exp):
    def __init__(self, args, trial, df_train, df_valid, df_test, use_pretrain, setting):
        super(Exp_Basic, self).__init__(args)
        self.args = args
        self.trial = trial
        self.path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.path = os.path.join(self.path, str(self.trial))
        os.makedirs(self.path, exist_ok=True)
            

        self.train_feature, self.train_label = self.process_data(df_train)
        # self.valid_feature, self.valid_label = self.process_data(df_valid)
        self.test_feature, self.test_label, self.df_test = self.process_data(df_test, test=True)
        self.model = self._build_model()


    def _build_model(self):
        model_dict = {
            'Linear': LinearRegression(),
            'Gbrt': HistGradientBoostingRegressor(max_iter=self.args.max_iter, max_depth=self.args.max_depth),
            'Rf': RandomForestRegressor(n_estimators=self.args.ntrees, max_depth=self.args.max_depth, min_samples_split=self.args.min_split, min_samples_leaf=self.args.min_leaf)
        }
        
        model = model_dict[self.args.model] 
    
        return model
    
    
    def process_data(self, df, test=False):
        gb = df.groupby("code")
        features = []
        labels = []
        df_list = []
        for name, group in gb:
            label = group['label'].to_numpy()
            feature = group.loc[:, ['open_norm', 'close_norm', 'high_norm', 'low_norm', 'return_norm', 'volume_norm', 'amount_norm']].to_numpy()
            l, d = feature.shape
            seq = self.args.seq_len
            features.extend([feature[i:i+seq, :] for i in range(l-seq+1)])
            if test:
                labels.extend(label[i+seq-1:i+seq] for i in range(l-seq+1))
                df_list.append(group.iloc[seq-1:])
            else:
                labels.extend(label[i+seq-self.args.train_label_len:i+seq] for i in range(l-seq+1))
        
        features = np.stack(features, axis=0)
        labels = np.stack(labels, axis=0)
        length = features.shape[0]

        if self.args.flat:
            features = features.reshape((length, -1))

        if test:
            return features, labels, pd.concat(df_list)

        return features, labels


    def train(self):
        self.model.fit(self.train_feature, self.train_label)

    def predict(self):
        prediction = self.model.predict(self.test_feature)
        return prediction.reshape(-1), self.test_label.reshape(-1)
    