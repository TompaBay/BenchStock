import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

# from data_provider.data_factory import data_provider
from exp.exp import Exp
from models import Adarnn
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


def get_index(num_domain=2):
    index = []
    for i in range(num_domain):
        for j in range(i+1, num_domain+1):
            index.append((i, j))
    return index


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


class Exp_Adarnn(Exp):
    def __init__(self, args, trial, df_train, df_valid, df_test, use_pretrain, setting):
        super(Exp_Adarnn, self).__init__(args)
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
            
        # dates = df_train['date'].unique()
        # dates.sort()
        # date_splits = np.array_split(np.array(dates), self.args.n_domains)

        # train_list = [df_train[(df_train['date'] <= date[-1]) & (df_train['date'] >= date[0])] for date in date_splits]
        # train_list = [self.process_data(train) for train in train_list]
        # input_dim = train_list[0][0][0].shape[1]
        # valid_feature, valid_label = self.process_data(df_valid)

        # train_list = [TrainDataset(train_feature, train_label) for (train_feature, train_label) in train_list]
        # train_args = dict(shuffle=True, batch_size=self.args.batch_size, num_workers=8)
        # self.train_loader_list = [DataLoader(train, **train_args) for train in train_list]

        # val = TrainDataset(valid_feature, valid_label)
        # val_args = dict(shuffle=False, batch_size=self.args.batch_size, num_workers=8)
        # self.val_loader = DataLoader(val, **val_args)

        # # if not full:
        # # test_feature, test_label = self.process_test_data(df_test)
        # # test = TrainDataset(test_feature, test_label)
        # # test_args = dict(shuffle=False, batch_size=1, num_workers=8)
        # # self.test_loader = DataLoader(test, **test_args)
        # # else:
        # test_feature, test_label, self.df_test = self.process_data(df_test, test=True)
        # test = TrainDataset(test_feature, test_label)
        # test_args = dict(shuffle=False, batch_size=self.args.batch_size, num_workers=8)
        # self.test_loader = DataLoader(test, **test_args)

        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.criterion_1 = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5)
        

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        if use_pretrain:
            # if self.full:
            #     checkpoint = torch.load( './' + file_name + '_full_us_model.pth')
            # else:

            checkpoint = torch.load(self.path + '/checkpoint.pth') 
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    def _build_model(self):
        model_dict = {
            'Adarnn': Adarnn
        }

        model = model_dict[self.args.model].Model(self.args).float()

        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        model.to(self.args.gpu)

        print("build model")
        print(self.args.gpu)
    
        return model

    # def _get_data(self, flag):
    #     data_set, data_loader = data_provider(self.args, flag)
    #     return data_set, data_loader
    
    
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


    def transform_type(self, init_weight):
        weight = torch.ones(self.args.e_layers, self.args.seq_len).to(self.args.gpu)
        for i in range(self.args.e_layers):
            for j in range(self.args.seq_len):
                weight[i, j] = init_weight[i][j].item()
        return weight


    def train(self):
        best_r2 = -np.inf
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        initial_memory = torch.cuda.memory_allocated()
        print("initial_memory:", initial_memory)
        dist_mat, weight_mat = None, None
        for i in range(self.args.train_epochs):
            print('Epoch', i + 1)
            loss, loss1, weight_mat, dist_mat = self.train_epoch(i, dist_mat, weight_mat)
            val_loss, val_r2 = self.vali()

            current_memory = torch.cuda.memory_allocated()
            print("current_memory:", current_memory)

            if val_r2 > best_r2:
                checkpoint = {
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict()
                             }
                
                torch.save(checkpoint, self.path + '/checkpoint.pth')
                # if self.full:
                #     torch.save(checkpoint,  './' + file_name + '_full_us_model.pth')
                # else:
                #     torch.save(checkpoint,  './' + file_name + '_us_model.pth') 

            self.scheduler.step()


    def train_epoch(self, epoch, dist_old=None, weight_mat=None):
        self.model.train()

        out_weight_list = None
        loss_all = []
        loss_1_all = []

        loss_ls = []
        loss_1_ls = []

        dates = self.df_train['date'].unique()
        dates.sort()

        years = np.arange(self.args.year, self.args.year + self.args.train_size + 1)
        
        print("Train")
        for i in range(len(years) - 1):
            start = max(bisect.bisect_left(dates, str(years[i]) + "-01-01") - self.args.seq_len, 0)
            df = self.df_train[(self.df_train['date'] >= dates[start]) & (self.df_train['date'] < (str(years[i+1]) + "-01-01"))]

            print("after dataloader")
            pid = psutil.Process()

            # Get memory usage statistics
            memory_info = pid.memory_info()

            memory_usage_gb = memory_info.rss / (1024 * 1024 * 1024)
            print("Memory usage (GB):", memory_usage_gb)

            dates = df['date'].unique()
            dates.sort()
            date_splits = np.array_split(np.array(dates), self.args.n_domains)

            train_list = [df[(df['date'] <= date[-1]) & (df['date'] >= date[0])] for date in date_splits]
            train_list = [self.process_data(train) for train in train_list]
            input_dim = train_list[0][0][0].shape[1]

            train_list = [TrainDataset(train_feature, train_label) for (train_feature, train_label) in train_list]
            train_args = dict(shuffle=True, batch_size=self.args.batch_size, num_workers=8)
            self.train_loader_list = [DataLoader(train, **train_args) for train in train_list]

            dist_mat = torch.zeros(self.args.e_layers, self.args.seq_len).to(self.args.gpu)
            len_loader = np.inf
            for loader in self.train_loader_list:
                if len(loader) < len_loader:
                    len_loader = len(loader)

            for data_all in tqdm(zip(*self.train_loader_list), total=len_loader):
                self.optimizer.zero_grad()
                list_feat = []
                list_label = []
                for data in data_all:
                    feature, label = data[0].to(self.args.gpu), data[1].to(self.args.gpu)
                    list_feat.append(feature)
                    list_label.append(label)
                    
                flag = False

                index = get_index(len(data_all) - 1)
                for temp_index in index:
                    s1 = temp_index[0]
                    s2 = temp_index[1]
                    if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                        flag = True
                        break
                if flag:
                    continue

                total_loss = torch.zeros(1, requires_grad=True).to(self.args.gpu)
                for i in range(len(index)):
                    feature_s = list_feat[index[i][0]]
                    feature_t = list_feat[index[i][1]]
                    label_reg_s = list_label[index[i][0]]
                    label_reg_t = list_label[index[i][1]]
                    feature_all = torch.cat((feature_s, feature_t), 0)

                    if epoch < self.args.pre_epoch:
                        pred_all, loss_transfer, out_weight_list = self.model.forward_pre_train(
                            feature_all, len_win=self.args.win_len)
                    else:
                        pred_all, loss_transfer, dist, weight_mat = self.model.forward_Boosting(
                            feature_all, weight_mat)
                        dist_mat = dist_mat + dist
                    pred_s = pred_all[0:feature_s.size(0)]
                    pred_t = pred_all[feature_s.size(0):]

                    loss_s = self.criterion(pred_s, label_reg_s)
                    loss_t = self.criterion(pred_t, label_reg_t)

                    loss_l1 = self.criterion_1(pred_s, label_reg_s)
                    total_loss = total_loss + loss_s + loss_t + self.args.dw * loss_transfer
                loss_all.append(
                    [total_loss.item(), (loss_s + loss_t).item(), loss_transfer.item()])
                loss_1_all.append(loss_l1.item())
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), 3.)
                self.optimizer.step()

            loss = np.array(loss_all).mean(axis=0)
            loss_l1 = np.array(loss_1_all).mean()
        
            loss_ls.append(loss)
            loss_1_ls.append(loss_l1)
        
        loss = np.array(loss_ls).mean()
        loss_l1 = np.array(loss_1_ls).mean()

        if epoch >= self.args.pre_epoch:
            if epoch > self.args.pre_epoch:
                weight_mat = self.model.update_weight_Boosting(
                    weight_mat, dist_old, dist_mat)
            return loss, loss_l1, weight_mat, dist_mat
        else:
            weight_mat = self.transform_type(out_weight_list)
            return loss, loss_l1, weight_mat, None
        
    
    def vali(self):
        self.model.eval()
        val_loss = 0
        prediction = []
        label = []
        loss_ls = []

        dates = self.df_valid['date'].unique()
        dates.sort()

        years = np.arange(self.args.year + self.args.train_size, self.args.year + self.args.train_size + self.args.val_size + 1)
        
        print("Valid")
        for i in range(len(years) - 1):
            val_loss = 0
            start = max(bisect.bisect_left(dates, str(years[i]) + "-01-01") - self.args.seq_len, 0)
            df = self.df_valid[(self.df_valid['date'] >= dates[start]) & (self.df_valid['date'] < (str(years[i+1]) + "-01-01"))]

            valid_feature, valid_label = self.process_data(df)

            pid = psutil.Process()

            # Get memory usage statistics
            memory_info = pid.memory_info()

            memory_usage_gb = memory_info.rss / (1024 * 1024 * 1024)
            print("after process")
            print("Memory usage (GB):", memory_usage_gb)

            valid = TrainDataset(valid_feature, valid_label)
            valid_args = dict(shuffle=False, batch_size=self.args.batch_size, num_workers=8)
            self.val_loader = DataLoader(valid, **valid_args)

            print("after dataloader")
            pid = psutil.Process()

            # Get memory usage statistics
            memory_info = pid.memory_info()

            memory_usage_gb = memory_info.rss / (1024 * 1024 * 1024)
            print("Memory usage (GB):", memory_usage_gb)

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
        # if self.args.data == "full":
        #     checkpoint = torch.load( './' + file_name + '_full_us_model.pth')
        # else:
        #     checkpoint = torch.load('./' + file_name + '_us_model.pth') 
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