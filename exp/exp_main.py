import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

# from data_provider.data_factory import data_provider
from exp.exp import Exp
from models import Lstm, Alstm, Crossformer, Dlinear, Nlinear, Mlp
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, r2
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import bisect
import psutil
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
        x = torch.FloatTensor(self.X[index])
        y = torch.FloatTensor(self.Y[index])
        return x, y


class Exp_Main(Exp):
    def __init__(self, args, trial, df_train, df_valid, df_test, use_pretrain, setting):
        super(Exp_Main, self).__init__(args)
        self.args = args
        self.trial = trial
        self.path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.path = os.path.join(self.path, str(self.trial))
        os.makedirs(self.path, exist_ok=True)

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5)

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        if use_pretrain:
            checkpoint = torch.load(self.path + '/checkpoint.pth') 
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    def _build_model(self):
        model_dict = {
            'Lstm': Lstm,
            'Alstm': Alstm,
            'Crossformer': Crossformer,
            'Dlinear': Dlinear,
            'Nlinear': Nlinear,
            'Mlp': Mlp
        }

        model = model_dict[self.args.model].Model(self.args).float()
        model.to(self.args.gpu)

        print("build model")
        print(self.args.gpu)
    
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
        best_r2 = -np.inf
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for i in range(self.args.train_epochs):
            print('Epoch', i + 1)
            self.train_epoch()
            val_loss, val_r2 = self.vali()

            if val_r2 > best_r2:
                checkpoint = {
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict()
                             }
                
                torch.save(checkpoint, self.path + '/checkpoint.pth')

            self.scheduler.step()


    def train_epoch(self):
        self.model.train()
        train_loss = 0
        prediction = []
        label = []

        dates = self.df_train['date'].unique()
        dates.sort()

        years = np.arange(self.args.year, self.args.year + self.args.train_size + 1)
        
        print("Train")
        for i in range(len(years) - 1):
            start = max(bisect.bisect_left(dates, str(years[i]) + "-01-01") - self.args.seq_len, 0)
            df = self.df_train[(self.df_train['date'] >= dates[start]) & (self.df_train['date'] < (str(years[i+1]) + "-01-01"))]

            train_feature, train_label = self.process_data(df)
            train = TrainDataset(train_feature, train_label)
            train_args = dict(shuffle=True, batch_size=self.args.batch_size, num_workers=8)
            self.train_loader = DataLoader(train, **train_args)

            for (inputs, targets) in tqdm(self.train_loader):
                inputs, targets = inputs.to(self.args.gpu), targets.to(self.args.gpu)
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.criterion(output, targets)

                if self.args.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                train_loss += loss.item()

                prediction.extend(list(output.detach().cpu().numpy().reshape(-1)))
                label.extend(list(targets.detach().cpu().numpy().reshape(-1)))

                del inputs
                del targets
                del loss
                
            train_loss /= len(self.train_loader)
            train_r2 = r2(prediction, label)
            print("Training r2:"+ str(train_r2) + "  Training loss:" + str(train_loss))
            
        del train_feature
        del train_label
        del train
        del self.train_loader
        del prediction
        del label

    
    def vali(self):
        self.model.eval()
        loss_ls = []
        prediction = []
        label = []
        

        dates = self.df_valid['date'].unique()
        dates.sort()

        years = np.arange(self.args.year + self.args.train_size, self.args.year + self.args.train_size + self.args.val_size + 1)
        
        print("Valid")
        for i in range(len(years) - 1):
            val_loss = 0
            start = max(bisect.bisect_left(dates, str(years[i]) + "-01-01") - self.args.seq_len, 0)
            df = self.df_valid[(self.df_valid['date'] >= dates[start]) & (self.df_valid['date'] < (str(years[i+1]) + "-01-01"))]

            valid_feature, valid_label = self.process_data(df)

            valid = TrainDataset(valid_feature, valid_label)
            valid_args = dict(shuffle=False, batch_size=self.args.batch_size, num_workers=8)
            self.val_loader = DataLoader(valid, **valid_args)

            for (inputs, targets) in tqdm(self.val_loader):
                inputs, targets = inputs.to(self.args.gpu), targets.to(self.args.gpu)
                
                output = self.model(inputs)
                loss = self.criterion(output, targets)
                val_loss += loss.item()

                prediction.extend(list(output.detach().cpu().numpy().reshape(-1)))
                label.extend(list(targets.detach().cpu().numpy().reshape(-1)))

                del inputs
                del targets
                del loss

            val_loss /= len(self.val_loader)
            loss_ls.append(val_loss)

        val_loss = np.mean(np.array(loss_ls))
        val_r2 = r2(prediction, label)
        print("Valid r2:"+ str(val_r2) + "  Valid loss:" + str(val_loss))

        del valid_feature
        del valid_label
        del valid
        del self.val_loader
        del prediction
        del label

        return val_loss, val_r2


    def predict(self):
        checkpoint = torch.load(self.path + '/checkpoint.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        prediction = []
        label = []

        test_feature, test_label, self.df_test = self.process_data(self.df_test, test=True)
        test = TrainDataset(test_feature, test_label)
        test_args = dict(shuffle=False, batch_size=self.args.batch_size, num_workers=8)
        self.test_loader = DataLoader(test, **test_args)

        for (inputs, targets) in tqdm(self.test_loader):
            inputs, targets = inputs.to(self.args.gpu), targets.to(self.args.gpu)
            output = self.model(inputs, test=True)
            
            prediction.append(output.detach().cpu().numpy().reshape(-1))
            label.append(targets.detach().cpu().numpy().reshape(-1))

            del inputs
            del targets

        return np.concatenate(prediction, axis=0), np.concatenate(label, axis=0)