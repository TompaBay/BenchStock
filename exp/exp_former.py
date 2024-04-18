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

        train_feature, train_stamp, train_label = self.process_data(df_train)
        valid_feature, valid_stamp, valid_label = self.process_data(df_valid)

        train = TrainDataset(train_feature, self.args.seq_len, self.args.label_len, self.args.pred_len, train_stamp, train_label)
        train_args = dict(shuffle=True, batch_size=self.args.batch_size, num_workers=8)
        self.train_loader = DataLoader(train, **train_args)

        val = TrainDataset(valid_feature, self.args.seq_len, self.args.label_len, self.args.pred_len, valid_stamp, valid_label)
        val_args = dict(shuffle=False, batch_size=self.args.batch_size, num_workers=8)
        self.val_loader = DataLoader(val, **val_args)

        # if not full:
        # test_feature, test_label = self.process_test_data(df_test)
        # test = TrainDataset(test_feature, test_label)
        # test_args = dict(shuffle=False, batch_size=1, num_workers=8)
        # self.test_loader = DataLoader(test, **test_args)
        # else:
        test_feature, test_stamp, test_label, self.df_test = self.process_data(df_test, test=True)
        test = TrainDataset(test_feature, self.args.seq_len, self.args.label_len, self.args.pred_len, test_stamp, test_label)
        test_args = dict(shuffle=False, batch_size=self.args.batch_size, num_workers=8)
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


    def _build_model(self):
        model_dict = {
            'Fedformer': Fedformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Logsparse': Logsparse
        }
        # device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
        model = model_dict[self.args.model].Model(self.args).float()

        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        model.to(self.args.gpu)
    
        return model

    # def _get_data(self, flag):
    #     data_set, data_loader = data_provider(self.args, flag)
    #     return data_set, data_loader
    
    
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
        # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.gpu)

        return outputs#, batch_y


    def train(self):
        # self.path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(self.path):
        #     os.makedirs(self.path)

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
                # if self.full:
                #     torch.save(checkpoint,  './' + file_name + '_full_us_model.pth')
                # else:
                #     torch.save(checkpoint,  './' + file_name + '_us_model.pth') 

            self.scheduler.step()


    def train_epoch(self):
        self.model.train()
        train_loss = 0
        prediction = []
        label = []

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
            break
            
        train_loss /= len(self.train_loader)
        train_r2 = r2(prediction, label)
        print("Training r2:"+ str(train_r2) + "  Training loss:" + str(train_loss))

    
    def vali(self):
        self.model.eval()
        val_loss = 0
        prediction = []
        label = []

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
            break

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



# class Exp_Former(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Former, self).__init__(args)

#     def _build_model(self):
#         model_dict = {
#             'Autoformer': Autoformer,
#             'Transformer': Transformer,
#             'Informer': Informer,
#             'Reformer': Reformer,
#         }
#         # device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
#         model = model_dict[self.args.model].Model(self.args).float()

#         # if self.args.use_multi_gpu and self.args.use_gpu:
#         #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         # model.to(self.args.gpu)
    
#         return model

#     def _get_data(self, flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader

#     def _select_optimizer(self):
#         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         return model_optim

#     def _select_criterion(self):
#         criterion = nn.MSELoss()
#         return criterion

#     def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
#         # decoder input
#         dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#         dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#         # encoder - decoder

#         def _run_model():
#             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#             if self.args.output_attention:
#                 outputs = outputs[0]
#             return outputs

#         if self.args.use_amp:
#             with torch.cuda.amp.autocast():
#                 outputs = _run_model()
#         else:
#             outputs = _run_model()

#         f_dim = -1 if self.args.features == 'MS' else 0
#         outputs = outputs[:, -self.args.pred_len:, f_dim:]
#         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

#         return outputs, batch_y

#     def vali(self, vali_data, vali_loader, criterion):
#         total_loss = []
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

#                 pred = outputs.detach().cpu()
#                 true = batch_y.detach().cpu()

#                 loss = criterion(pred, true)

#                 total_loss.append(loss)
#         total_loss = np.average(total_loss)
#         self.model.train()
#         return total_loss

#     def train(self, setting):
#         train_data, train_loader = self._get_data(flag='train')
#         vali_data, vali_loader = self._get_data(flag='val')
#         test_data, test_loader = self._get_data(flag='test')

#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         time_now = time.time()

#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

#         model_optim = self._select_optimizer()
#         criterion = self._select_criterion()

#         if self.args.use_amp:
#             scaler = torch.cuda.amp.GradScaler()

#         for epoch in range(self.args.train_epochs):
#             iter_count = 0
#             train_loss = []

#             self.model.train()
#             epoch_time = time.time()
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
#                 iter_count += 1
#                 model_optim.zero_grad()
#                 batch_x = batch_x.float().to(self.device)

#                 batch_y = batch_y.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

#                 loss = criterion(outputs, batch_y)
#                 train_loss.append(loss.item())

#                 if (i + 1) % 100 == 0:
#                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()

#                 if self.args.use_amp:
#                     scaler.scale(loss).backward()
#                     scaler.step(model_optim)
#                     scaler.update()
#                 else:
#                     loss.backward()
#                     model_optim.step()

#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)
#             vali_loss = self.vali(vali_data, vali_loader, criterion)
#             test_loss = self.vali(test_data, test_loader, criterion)

#             print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
#                 epoch + 1, train_steps, train_loss, vali_loss, test_loss))
#             early_stopping(vali_loss, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break

#             adjust_learning_rate(model_optim, epoch + 1, self.args)

#         best_model_path = path + '/' + 'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))

#         return

#     def test(self, setting, test=0):
#         test_data, test_loader = self._get_data(flag='test')
#         if test:
#             print('loading model')
#             self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

#         preds = []
#         trues = []
#         folder_path = './test_results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

#                 outputs = outputs.detach().cpu().numpy()
#                 batch_y = batch_y.detach().cpu().numpy()

#                 pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
#                 true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

#                 preds.append(pred)
#                 trues.append(true)
#                 if i % 20 == 0:
#                     input = batch_x.detach().cpu().numpy()
#                     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

#         preds = np.concatenate(preds, axis=0)
#         trues = np.concatenate(trues, axis=0)
#         print('test shape:', preds.shape, trues.shape)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#         print('test shape:', preds.shape, trues.shape)

#         # result save
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         mae, mse, rmse, mape, mspe = metric(preds, trues)
#         print('mse:{}, mae:{}'.format(mse, mae))
#         f = open("result.txt", 'a')
#         f.write(setting + "  \n")
#         f.write('mse:{}, mae:{}'.format(mse, mae))
#         f.write('\n')
#         f.write('\n')
#         f.close()

#         np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
#         np.save(folder_path + 'pred.npy', preds)
#         np.save(folder_path + 'true.npy', trues)

#         return

#     def predict(self, setting, load=False):
#         pred_data, pred_loader = self._get_data(flag='pred')

#         if load:
#             path = os.path.join(self.args.checkpoints, setting)
#             best_model_path = path + '/' + 'checkpoint.pth'
#             logging.info(best_model_path)
#             self.model.load_state_dict(torch.load(best_model_path))

#         preds = []

#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

#                 pred = outputs.detach().cpu().numpy()  # .squeeze()
#                 preds.append(pred)

#         preds = np.array(preds)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

#         # result save
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         np.save(folder_path + 'real_prediction.npy', preds)

#         return
