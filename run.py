import psutil
import argparse
import os
import json
import torch
import pickle
import bisect
import numpy as np
import pandas as pd
from tqdm import tqdm

from exp.exp_basic import Exp_Basic
from exp.exp_main import Exp_Main
from exp.exp_former import Exp_Former
from exp.exp_graph import Exp_Graph
from exp.exp_vae import Exp_Vae
from exp.exp_adarnn import Exp_Adarnn
from utils.metrics import sharpe_ratio, r2, max_drawdown
from backtest.backtest import Backtest


def main():
    parser = argparse.ArgumentParser(description='Benchmark for Stock Price Prediction')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name')
    parser.add_argument('--exp', type=str, required=True, default='Main',
                        help='train function name, options: [Basic, Main, Former, Graph, Vae, Adarnn]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--market', type=str, required=True, default='us', help="data market")
    parser.add_argument('--full', type=str, required=True, default="True", help='use full dataset or not')
    parser.add_argument('--graph', type=str, default="graph", help="Use graph of hypergraph in graph networks, options: [graph, hypergraph]")
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='us.feather', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # dataset split
    parser.add_argument('--start', type=int, default=1960, help="The whole trial's start year")
    parser.add_argument('--train_start', type=int, default=2000, help='training start year')
    parser.add_argument('--train_size', type=int, default=5, help='number of years included in training set')
    parser.add_argument('--val_start_year', type=int, default=2005, help='validation start year')
    parser.add_argument('--val_size', type=int, default=3, help='number of years included in validation set')
    parser.add_argument('--test_start_year', type=int, default=2008, help='test start year')
    parser.add_argument('--test_size', type=int, default=16, help='number of years included in test set')
    parser.add_argument('--test_end_year', type=int, default=2023, help='test start year')


    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--train_label_len', type=int, default=48, help='start token length')

    # model define
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--final_out', type=int, default=1, help='final output size (Only used in Exp_Former)')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--sparse_flag', type=bool, default=False, help='Whether to apply logsparse mask for LogSparse Transformer')
    parser.add_argument('--win_len', type=int, default=6, help='Local attention length for LogSparse Transformer or length window of Adarnn')
    parser.add_argument('--res_len', type=int, default=None, help='Restart attention length for LogSparse Transformer')
    parser.add_argument('--qk_ker', type=int, default=4, help='Key/Query convolution kernel length for LogSparse Transformer')
    parser.add_argument('--v_conv', type=int, default=0, help='Whether to apply ConvAttn for values (in addition to K/Q for LogSparseAttn')
    parser.add_argument('--win_size', type=int, default=2, help='Merging every window size of adjacent segments in Crossformer')
    parser.add_argument('--seg_len', type=int, default=2, help='Segment length for Crossformer')
    parser.add_argument('--hidden_layer', default=[64, 32, 16], nargs='+', type=int,
                        help="number of hidden units in each layer in MLP and Adarnn, a standard format of the input would be like '128 64 32', which standards for three hidden layers")
    parser.add_argument('--flat', type=str, default="False", help="Whether the sequence data should be transformed into flatten form")
    parser.add_argument('--ntrees', type=int, default=10, help="Number of trees for random forest model")
    parser.add_argument('--max_iter', type=int, default=50, help="Maximum iteration of gradient boost model")
    parser.add_argument('--max_depth', type=int, default=10, help="Maximum depth of tree models")
    parser.add_argument('--min_split', type=int, default=10000, help="Minimum samples required for a split in tree model")
    parser.add_argument('--min_leaf', type=int, default=10000, help="Minimum samples within a leaf in tree models")
    parser.add_argument('--n_domains', type=int, default=2, help="Number of domains within Adarnn")
    parser.add_argument('--trans_loss', type=str, default='adv', help="Type of loss to calculate losses between domains")
    parser.add_argument('--use_bottleneck', type=bool, default=True, help="Use bottleneck network in Adarnn")
    parser.add_argument('--bottleneck_width', type=int, default=64, help="bottleneck width of bottleneck network in Adarnn")
    parser.add_argument('--model_type', type=str, default='AdaRNN', help="Type of model used in Adarnn")
    parser.add_argument('--pre_epoch', type=int, default=10, help="Epochs before applying boost training in Adarnn")
    parser.add_argument('--dw', type=float, default=0.5, help="Weight of transfer loss in Adarnn")

    # supplementary config for FEDformer model
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')
    parser.add_argument('--gl', type=float, default=0, help='weight of graph loss in Alsp-tf')
    parser.add_argument('--alpha', type=float, default=0, help='weight of rank loss in graph networks')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    #  backtest
    parser.add_argument('-topk', default=50, type=int, help='topk')
    parser.add_argument('-fee', default=0.0035, type=float, help='Trading fee per share')
    parser.add_argument('-cash', default=100000000, type=float, help='Initial Capital')
    parser.add_argument('-turnover', default=5, type=int, help='Number of stocks to change per day')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=9, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # if args.use_gpu and args.use_multi_gpu:
    #     args.devices = args.devices.replace(' ', '')
    #     device_ids = args.devices.split(',')
    #     args.device_ids = [int(id_) for id_ in device_ids]
    #     args.gpu = args.device_ids[0]
    args.gpu = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    print('Args in experiment:')
    print(args)
    if args.full == 'True':
        args.full = True
    else:
        args.full = False

    if args.flat == 'True':
        args.flat = True 
    else:
        args.flat = False
    
    exp_dict = {
        "Basic": Exp_Basic,
        "Main": Exp_Main,
        "Former": Exp_Former,
        "Graph": Exp_Graph,
        "Vae": Exp_Vae,
        "Adarnn": Exp_Adarnn
    }
    
    Exp = exp_dict[args.exp]

    df = pd.read_feather(os.path.join(args.root_path, args.data_path))

    print(df)
    # Get the current process ID
    pid = psutil.Process()

    # Get memory usage statistics
    memory_info = pid.memory_info()

    # Print memory usage in bytes
    print("Memory usage (bytes):", memory_info.rss)

    # Convert memory usage to human-readable format (e.g., MB)
    memory_usage_mb = memory_info.rss / (1024 * 1024)
    print("Memory usage (MB):", memory_usage_mb)

    memory_usage_gb = memory_info.rss / (1024 * 1024 * 1024)
    print("Memory usage (GB):", memory_usage_gb)
        
    if args.market == 'us':
        df.rename(columns={'PERMNO': 'code'}, inplace=True)
        df.drop(columns=['return', 'high', 'low',
       'open', 'close', 'volume', 'amount'], inplace=True)
        

    df.sort_values(by=['code', 'date'], inplace=True)
    date_ls = list(df['date'].unique())
    date_ls.sort()

    save_path = "./backtest/" + args.model
    os.makedirs(save_path, exist_ok=True)
    print("save_path:", save_path)

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            
            os.makedirs(save_path + "/" + str(ii + 1), exist_ok=True)
            performance = {}
            for year in range(args.train_start, args.train_start + args.test_size):
                # train_year = year
                args.year = year
                val_year = year + args.train_size
                test_year = year + args.train_size + args.val_size

                train_start = date_ls[max(bisect.bisect_left(date_ls, str(args.train_start) + "-01-01") - args.seq_len, 0)]
                valid_start = date_ls[bisect.bisect_left(date_ls, str(val_year) + "-01-01") - args.seq_len]
                test_start = date_ls[bisect.bisect_left(date_ls, str(test_year) + "-01-01") - args.seq_len]

                df_train = df[(df['date'] >= train_start) & (df['date'] < (str(val_year) + "-01-01"))]
                df_valid = df[(df['date'] >= valid_start) & (df['date'] < (str(test_year) + "-01-01"))]
                df_test = df[(df['date'] >= test_start) & (df['date'] < (str(test_year + 1) + "-01-01"))]

                # Get the current process ID
                pid = psutil.Process()

                # Get memory usage statistics
                memory_info = pid.memory_info()

                # Print memory usage in bytes
                print("Memory usage (bytes):", memory_info.rss)

                # Convert memory usage to human-readable format (e.g., MB)
                memory_usage_mb = memory_info.rss / (1024 * 1024)
                print("Memory usage (MB):", memory_usage_mb)

                memory_usage_gb = memory_info.rss / (1024 * 1024 * 1024)
                print("Memory usage (GB):", memory_usage_gb)

                # Select stocks with enough test data
                count = df_test.groupby("code").count()
                # if not args.full:
                #     mode_num = count['code'].mode().to_numpy()[0]
                #     count = count[count['code'] == mode_num]
                # else:
                #     count = count[count['code'] >= args.l]
                count = count[count['date'] > args.seq_len]
                
                df_test = df_test[df_test['code'].isin(list(count.index))]
                df_test.sort_values(by=['code', 'date'], inplace=True)
                test_dates = df_test['date'].unique()
                test_dates.sort()
                test_dates = test_dates[args.seq_len - 1:]
                if args.exp == "Former":
                    test_dates = test_dates[:-args.pred_len]
                code_list = df_test['code'].unique()

                use_pretrain = True
                if year == args.start:
                    use_pretrain = False

                exp = Exp(args, ii, df_train, df_valid, df_test, use_pretrain, setting)
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train()
                
                performance[test_year] = {}
                account = Backtest(args.cash, topk=args.topk, n_drop=args.turnover, trade_cost=args.fee)

                print(args.full)
                if args.full:
                    prediction, label = exp.predict()
                    df_test = exp.df_test
                    print(df_test)
                    df_test['pred'] = prediction
                    print(df_test)
                    for j in tqdm(range(len(test_dates))):
                        date = test_dates[j]
                        df_today = df_test[df_test['date'] == date].loc[:, ['code', 'pred', 'date', 'label', 'DlyPrc', 'ShrOut', 'DlyVol', 'SecurityEndDt']]
                        account.trade_us(df_today, args.full)
                else:
                    prediction, label = exp.predict()
                    for j in tqdm(range(len(test_dates))):
                        date = test_dates[j]
                        df_today = df_test[df_test['date'] == date]
                        df_today = df_today.loc[:, ['code', 'date', 'label', 'DlyPrc', 'ShrOut', 'DlyVol']]
                        df_pred = pd.DataFrame({'code': code_list, 'pred': prediction[j]})
                        account.trade_us(df_pred.merge(df_today, on="code", how="left"))

                torch.cuda.empty_cache()

                value = account.cum_value / args.cash
                values = account.values 
                
                returns = [values[i + 1] / values[i] - 1 for i in range(len(values) - 1)]     
                print("value:" + str(value))
                
                performance[test_year]['IRR'] = float(value) - 1
                performance[test_year]['SR'] = float(sharpe_ratio(returns, value))
                performance[test_year]['MDD'] =  max_drawdown(values)
                performance[test_year]['Loss'] = float(np.mean((label - prediction) ** 2))
                performance[test_year]['r2'] = float(r2(prediction.reshape(-1), label.reshape(-1)))

                print(performance)

                with open(save_path + "/" + str(ii+1) + "/" + str(test_year) + '.pkl', 'wb') as file:
                    pickle.dump(account, file)


                try:
                    with open(save_path + "/" + str(ii+1) + "/performance.json", 'r') as file:
                        data = json.load(file)
                except FileNotFoundError:
                    data = {}
                
                data[test_year] = performance[test_year]
                    
                with open(save_path + "/" + str(ii+1) + "/performance.json" , 'w') as file:
                    json.dump(data, file)
                

            # print(performance)


            # with open(save_path + "/" + str(ii+1) + "/performance.json" , 'w') as file:
            #     json.dump(performance, file)

            # exp = Exp(args)  # set experiments
            # print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # exp.train(setting)

            # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test(setting)

            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)

            # torch.cuda.empty_cache()
    # else:
    #     ii = 0
    #     setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
    #                                                                                                   args.model,
    #                                                                                                   args.data,
    #                                                                                                   args.features,
    #                                                                                                   args.seq_len,
    #                                                                                                   args.label_len,
    #                                                                                                   args.pred_len,
    #                                                                                                   args.d_model,
    #                                                                                                   args.n_heads,
    #                                                                                                   args.e_layers,
    #                                                                                                   args.d_layers,
    #                                                                                                   args.d_ff,
    #                                                                                                   args.factor,
    #                                                                                                   args.embed,
    #                                                                                                   args.distil,
    #                                                                                                   args.des, ii)

    #     exp = Exp(args)  # set experiments
    #     print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #     exp.test(setting, test=1)
    #     torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
