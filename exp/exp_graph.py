import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

# from data_provider.data_factory import data_provider
from exp.exp import Exp
from models import Alsp_tf, Gat, Sthan_sr
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, r2

import numpy as np
import pandas as pd
from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric import utils

import os
import time
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)

    # Create a matrix to store DTW values
    dtw_matrix = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    
    dtw_matrix[0, 0] = 0
    
    # Calculate DTW values
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(s1[i - 1] - s2[j - 1])  # Euclidean distance as cost
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
    
    return dtw_matrix[n, m]



def DTW(data):
    # Calculate DTW distance between each pair
    num_sequences = data.shape[0]
    dtw_distances = np.zeros((num_sequences, num_sequences))

    for i in range(num_sequences):
        for j in range(num_sequences):
            dtw_distances[i, j] = dtw_distance(data[i], data[j])
    np.save("dtw.npy", dtw_distance)

    return dtw_distances


def mask(matrix, n):
    m = matrix < n
    matrix[m] = 1.0
    matrix[~m] = 0.0
    return matrix


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


class Exp_Graph(Exp):
    def __init__(self, args, trial, df_train, df_valid, df_test, use_pretrain, setting):
        super(Exp_Graph, self).__init__(args)
        self.args = args
        self.trial = trial
        self.path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.path = os.path.join(self.path, str(self.trial))
        os.makedirs(self.path, exist_ok=True)
        
        self._load_graph()
            
        train_feature, train_label = self.process_data(df_train)
        valid_feature, valid_label = self.process_data(df_valid)

        self.adj = mask(DTW(train_feature[0]), self.args.seq_len / 2)
        np.save("dynamic_graph.npy", self.adj)
        a=3/0


        train = TrainDataset(train_feature, train_label)
        train_args = dict(shuffle=True, batch_size=1, num_workers=8)
        self.train_loader = DataLoader(train, **train_args)

        val = TrainDataset(valid_feature, valid_label)
        val_args = dict(shuffle=False, batch_size=1, num_workers=8)
        self.val_loader = DataLoader(val, **val_args)

        test_feature, test_label = self.process_data(df_test)
        test = TrainDataset(test_feature, test_label)
        test_args = dict(shuffle=False, batch_size=1, num_workers=8)
        self.test_loader = DataLoader(test, **test_args)

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
            # if self.full:
            #     checkpoint = torch.load( './' + file_name + '_full_us_model.pth')
            # else:

            checkpoint = torch.load(self.path + '/checkpoint.pth') 
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    
    def _load_graph(self):
        self.adj_matrix = np.load("./dataset/" + self.args.market + "_" + self.args.graph + ".npy")
        self.edge = None
        if self.args.graph == "hypergraph":
            adj_sparse = sparse.coo_martix(self.adj_matrix)
            self.edge = utils.from_scipy_sparse_matrix(adj_sparse)[0].to(self.args.gpu)
        self.adj_matrix = torch.FloatTensor(self.adj_matrix).to(self.args.gpu)


    def _build_model(self):
        model_dict = {
            'Gat': Gat,
            'Sthan_sr': Sthan_sr,
            'Alsp_tf': Alsp_tf
        }
        # device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
        model = model_dict[self.args.model].Model(self.args).float()

        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        model.to(self.args.gpu)
    
        return model


    def rank_loss(self, output, target):
        shape = output.shape
        mse_loss = torch.mean(torch.pow(target - output, 2))
        all_ones = torch.ones(shape).to(self.args.gpu)
        pre_pw_dif =  (torch.matmul(output, torch.transpose(all_ones, 0, 1)) 
                    - torch.matmul(all_ones, torch.transpose(output, 0, 1)))
        gt_pw_dif = (
            torch.matmul(all_ones, torch.transpose(target,0,1)) -
            torch.matmul(target, torch.transpose(all_ones, 0,1))
        )
        rank_loss = torch.mean(F.relu(pre_pw_dif*gt_pw_dif))
        return mse_loss + self.args.alpha * rank_loss


    def graph_loss(self, output, adj_mtx):
        o = torch.log(torch.sigmoid(torch.mm(output, output.T)))
        no = torch.log(torch.sigmoid(torch.mm(-output, output.T)))
        neighbor = torch.diagonal(torch.mm(adj_mtx, o) * torch.eye(output.shape[0]).to(self.args.gpu))
        non_neighbor = torch.diagonal(torch.mm(1 - adj_mtx, no) * torch.eye(output.shape[0]).to(self.args.gpu))
        loss = -neighbor - non_neighbor
        return torch.mean(loss)
    

    def process_data(self, df):
        gb = df.groupby("code")
        features = []
        labels = []
        for name, group in gb:
            f_list = []
            l_list = []
            label = group['label'].to_numpy()
            feature = group.loc[:, ['open_norm', 'close_norm', 'high_norm', 'low_norm', 'return_norm', 'volume_norm', 'amount_norm']].to_numpy()
            l, d = feature.shape

            for i in range(l - self.args.seq_len + 1):
                f_list.append(feature[i : i+self.args.seq_len, :])
                l_list.append(label[i + self.args.seq_len - 1 : i + self.args.seq_len])

            feature = np.stack(f_list, axis=0)
            label = np.stack(l_list, axis=0)          

            features.append(feature)
            labels.append(label)

        return np.stack(features, axis=0).transpose((1, 0, 2, 3)), np.stack(labels, axis=0).transpose((1, 0, 2))


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
        for (inputs, targets) in tqdm(self.train_loader):
            inputs, targets = inputs.to(self.args.gpu).squeeze(0), targets.to(self.args.gpu).squeeze(0)
            self.optimizer.zero_grad()
            output = self.model(inputs, self.adj_matrix, self.edge)
    
            if self.args.gl != 0.0:
                loss = self.rank_loss(output[1], targets) + self.args.gl * self.graph_loss(output[0], self.adj)
            else:
                loss = self.rank_loss(output, targets)

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

    
    def vali(self):
        self.model.eval()
        val_loss = 0
        prediction = []
        label = []

        for (inputs, targets) in tqdm(self.val_loader):
            inputs, targets = inputs.to(self.args.gpu).squeeze(0), targets.to(self.args.gpu).squeeze(0)
            output = self.model(inputs, self.adj_matrix, self.edge)

            if self.args.gl != 0.0:
                loss = self.rank_loss(output[1], targets) + self.args.gl * self.graph_loss(output[0], self.adj)
            else:
                loss = self.rank_loss(output, targets)
            val_loss += loss.item()

            prediction.extend(list(output.detach().cpu().numpy().reshape(-1)))
            label.extend(list(targets.detach().cpu().numpy().reshape(-1)))

            del inputs
            del targets
            del loss

        val_loss /= len(self.val_loader)
        val_r2 = r2(prediction, label)
        print("Valid r2:"+ str(val_r2) + "  Valid loss:" + str(val_loss))
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

        for (inputs, targets) in tqdm(self.test_loader):
            inputs, targets = inputs.to(self.args.gpu).squeeze(0), targets.to(self.args.gpu).squeeze(0)
            output = self.model(inputs, self.adj_matrix, self.edge)
            
            prediction.append(output.detach().cpu().numpy().reshape(-1))
            label.append(targets.detach().cpu().numpy().reshape(-1))

            del inputs
            del targets

        return np.concatenate(prediction, axis=0), np.concatenate(label, axis=0)