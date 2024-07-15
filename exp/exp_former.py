import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

from data_provider.data_factory import data_provider
from exp.exp import Exp
from models import Informer, Autoformer, Transformer, Reformer, Logsparse, Fedformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, r2
from utils.timefeatures import time_features

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import bisect
import psutil
import os
import time
from tqdm import tqdm

import warnings
import numpy as np

warnings.filterwarnings('ignore')



class TrainDataset(Dataset):
    # load the dataset
    def __init__(self, x, seq_len, label_len, pred_len, data_stamp, data_label):
        self.X = x
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.data_stamp = data_stamp
        self.data_label = data_label

    # get number of items/rows in dataset
    def __len__(self):
        return len(self.X)

    # get row item at some index
    def __getitem__(self, index):
        s_end = self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.X[index, :s_end]
        seq_y = self.X[index, r_begin:r_end]
        seq_x_mark = self.data_stamp[index, :s_end]
        seq_y_mark = self.data_stamp[index, r_begin:r_end]
        seq_label = self.data_label[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_label
        


class Exp_Former(Exp):
    def __init__(self, args, trial, df_train, df_valid, df_test, use_pretrain, setting):
        super(Exp_Former, self).__init__(args)
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
            'Fedformer': Fedformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Logsparse': Logsparse
        }
        model = model_dict[self.args.model].Model(self.args).float()
        model.to(self.args.gpu)
    
        return model
    
    
    def process_data(self, df, test=False):
        gb = df.groupby("code")
        features = []
        stamps = []
        labels = []
        df_list = []
        
        for name, group in gb:
            label = group['label'].to_numpy()
            data_stamp = time_features(pd.to_datetime(group['date'].values), freq=self.args.freq)
            data_stamp = data_stamp.transpose(1, 0)
            
            feature = group.loc[:, ['open_norm', 'close_norm', 'high_norm', 'low_norm', 'return_norm', 'volume_norm', 'amount_norm']].to_numpy()
            l, d = feature.shape
            seq = self.args.seq_len
            features.extend([feature[i:i+seq+self.args.pred_len, :] for i in range(l - (seq + self.args.pred_len) + 1)])
            stamps.extend([data_stamp[i:i+seq+self.args.pred_len, :] for i in range(l - (seq + self.args.pred_len) + 1)])
  
            if test:
                labels.extend(label[i+seq-self.args.pred_len:i+seq] for i in range(l - (seq + self.args.pred_len) + 1))
                df_list.append(group.iloc[seq-1:-self.args.pred_len])
            else:
                labels.extend(label[i+seq-self.args.train_label_len:i+seq] for i in range(l-(seq + self.args.pred_len)+1))

        if test:
            return np.stack(features, axis=0), np.stack(stamps, axis=0), np.stack(labels, axis=0), pd.concat(df_list)
        
        return np.stack(features, axis=0), np.stack(stamps, axis=0), np.stack(labels, axis=0)


    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.gpu)
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:].squeeze(1)

        return outputs


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

            train_feature, train_stamp, train_label = self.process_data(df)

            train = TrainDataset(train_feature, self.args.seq_len, self.args.label_len, self.args.pred_len, train_stamp, train_label)
            train_args = dict(shuffle=True, batch_size=self.args.batch_size, num_workers=8)
            self.train_loader = DataLoader(train, **train_args)

            for (batch_x, batch_y, batch_x_mark, batch_y_mark, target) in tqdm(self.train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, target  = \
                    batch_x.float().to(self.args.gpu), batch_y.float().to(self.args.gpu), batch_x_mark.float().to(self.args.gpu), batch_y_mark.float().to(self.args.gpu), target.float().to(self.args.gpu)
                self.optimizer.zero_grad()
                output = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = self.criterion(output, target)

                if self.args.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item()

                prediction.extend(list(output.detach().cpu().numpy().reshape(-1)))
                label.extend(list(target.detach().cpu().numpy().reshape(-1)))

                del batch_x
                del batch_y
                del batch_x_mark
                del batch_y_mark
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

            valid_feature, valid_stamp, valid_label = self.process_data(df)

            val = TrainDataset(valid_feature, self.args.seq_len, self.args.label_len, self.args.pred_len, valid_stamp, valid_label)
            val_args = dict(shuffle=False, batch_size=self.args.batch_size, num_workers=8)
            self.val_loader = DataLoader(val, **val_args)

        for (batch_x, batch_y, batch_x_mark, batch_y_mark, target) in tqdm(self.val_loader):
            batch_x, batch_y, batch_x_mark, batch_y_mark, target = \
                batch_x.float().to(self.args.gpu), batch_y.float().to(self.args.gpu), batch_x_mark.float().to(self.args.gpu), batch_y_mark.float().to(self.args.gpu), target.float().to(self.args.gpu)
            output = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = self.criterion(output, target)
            val_loss += loss.item()

            prediction.extend(list(output.detach().cpu().numpy().reshape(-1)))
            label.extend(list(target.detach().cpu().numpy().reshape(-1)))

            del batch_x
            del batch_y
            del batch_x_mark
            del batch_y_mark
            del loss

            val_loss /= len(self.val_loader)
            loss_ls.append(val_loss)

        val_loss = np.mean(np.array(loss_ls))
        val_r2 = r2(prediction, label)
        print("Valid r2:"+ str(val_r2) + "  Valid loss:" + str(val_loss))

        del valid_feature
        del valid_stamp
        del valid_label
        del val
        del self.val_loader
        del prediction
        del label

        return val_loss, val_r2


    def predict(self):
        checkpoint = torch.load(self.path + '/checkpoint.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        test_feature, test_stamp, test_label, self.df_test = self.process_data(self.df_test, test=True)
        test = TrainDataset(test_feature, self.args.seq_len, self.args.label_len, self.args.pred_len, test_stamp, test_label)
        test_args = dict(shuffle=False, batch_size=self.args.batch_size, num_workers=8)
        self.test_loader = DataLoader(test, **test_args)

        prediction = []
        label = []

        for (batch_x, batch_y, batch_x_mark, batch_y_mark, target) in tqdm(self.test_loader):
            batch_x, batch_y, batch_x_mark, batch_y_mark, target = \
                batch_x.float().to(self.args.gpu), batch_y.float().to(self.args.gpu), batch_x_mark.float().to(self.args.gpu), batch_y_mark.float().to(self.args.gpu), target.float().to(self.args.gpu)
            output = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
            
            prediction.append(output.detach().cpu().numpy().reshape(-1))
            label.append(target.detach().cpu().numpy().reshape(-1))

            del batch_x
            del batch_y
            del batch_x_mark
            del batch_y_mark

        return np.concatenate(prediction, axis=0), np.concatenate(label, axis=0)


