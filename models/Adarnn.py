import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss import adv_loss, coral, kl_js, mmd, mutual_info, cosine, pair_dist


class TransferLoss(object):
    def __init__(self, loss_type='cosine', input_dim=512, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Supported loss_type: mmd(mmd_lin), mmd_rbf, coral, cosine, kl, js, mine, adv
        """
        self.loss_type = loss_type
        self.input_dim = input_dim
        self.device = device

    def compute(self, X, Y):
        """Compute adaptation loss

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix

        Returns:
            [tensor] -- transfer loss
        """
        if self.loss_type == 'mmd_lin' or self.loss_type =='mmd':
            mmdloss = mmd.MMD_loss(kernel_type='linear')
            loss = mmdloss(X, Y)
        elif self.loss_type == 'coral':
            loss = coral.CORAL(X, Y)
        elif self.loss_type == 'cosine' or self.loss_type == 'cos':
            loss = 1 - cosine.cosine(X, Y)
        elif self.loss_type == 'kl':
            loss = kl_js.kl_div(X, Y)
        elif self.loss_type == 'js':
            loss = kl_js.js(X, Y)
        elif self.loss_type == 'mine':
            mine_model = mutual_info.Mine_estimator(
                input_dim=self.input_dim, hidden_dim=60).cuda()
            loss = mine_model(X, Y)
        elif self.loss_type == 'adv':
            loss = adv_loss.adv(X, Y, input_dim=self.input_dim, hidden_dim=32, device=self.device)
        elif self.loss_type == 'mmd_rbf':
            mmdloss = mmd.MMD_loss(kernel_type='rbf')
            loss = mmdloss(X, Y)
        elif self.loss_type == 'pairwise':
            pair_mat = pair_dist.pairwise_dist(X, Y)
            import torch
            loss = torch.norm(pair_mat)

        return loss




class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.use_bottleneck = args.use_bottleneck
        self.bottleneck_width = args.bottleneck_width
        self.n_input = args.enc_in
        self.hiddens = args.hidden_layer
        self.num_layers = len(self.hiddens)
        self.n_output = args.c_out
        self.model_type = args.model_type
        self.trans_loss = args.trans_loss
        self.len_seq = args.seq_len
        self.device = args.gpu
        in_size = self.n_input

        features = nn.ModuleList()
        for hidden in self.hiddens:
            rnn = nn.GRU(
                input_size=in_size,
                num_layers=self.num_layers,
                hidden_size=hidden,
                batch_first=True,
                dropout=args.dropout
            )
            features.append(rnn)
            in_size = hidden
        self.features = nn.Sequential(*features)

        if args.use_bottleneck == True: 
            self.bottleneck = nn.Sequential(
                nn.Linear(self.hiddens[-1], self.bottleneck_width),
                nn.Linear(self.bottleneck_width, self.bottleneck_width),
                nn.BatchNorm1d(self.bottleneck_width),
                nn.ReLU(),
                nn.Dropout(),
            )
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            self.bottleneck[1].weight.data.normal_(0, 0.005)
            self.bottleneck[1].bias.data.fill_(0.1)
            self.fc = nn.Linear(self.bottleneck_width, self.n_output)
            torch.nn.init.xavier_normal_(self.fc.weight)
        else:
            self.fc_out = nn.Linear(self.hiddens[-1], self.n_output)

        # if self.model_type == 'AdaRNN':
        gate = nn.ModuleList()
        for i in range(self.num_layers):
            gate_weight = nn.Linear(
                self.len_seq * self.hiddens[i]*2, self.len_seq)
            gate.append(gate_weight)
        self.gate = gate

        bnlst = nn.ModuleList()
        for i in range(self.num_layers):
            bnlst.append(nn.BatchNorm1d(self.len_seq))
        self.bn_lst = bnlst
        self.softmax = torch.nn.Softmax(dim=0)
        self.init_layers()

    def init_layers(self):
        for i in range(len(self.hiddens)):
            self.gate[i].weight.data.normal_(0, 0.05)
            self.gate[i].bias.data.fill_(0.0)

    def forward_pre_train(self, x, len_win=0):
        out = self.gru_features(x)
        fea = out[0]
        if self.use_bottleneck == True:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()

        out_list_all, out_weight_list = out[1], out[2]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).to(self.device)
        for i in range(len(out_list_s)):
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2], device=self.device)
            h_start = 0 
            for j in range(h_start, self.len_seq, 1):
                i_start = j - len_win if j - len_win >= 0 else 0
                i_end = j + len_win if j + len_win < self.len_seq else self.len_seq - 1
                for k in range(i_start, i_end + 1):
                    weight = out_weight_list[i][j] if self.model_type == 'AdaRNN' else 1 / (
                        self.len_seq - h_start) * (2 * len_win + 1)
                    loss_transfer = loss_transfer + weight * criterion_transder.compute(
                        out_list_s[i][:, j, :], out_list_t[i][:, k, :])
        return fc_out, loss_transfer, out_weight_list

    def gru_features(self, x, predict=False):
        x_input = x
        out = None
        out_lis = []
        out_weight_list = [] if (
             self.model_type == 'AdaRNN') else None
        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float())
            x_input = out
            out_lis.append(out)
            if self.model_type == 'AdaRNN' and predict == False:
                out_gate = self.process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)
        return out, out_lis, out_weight_list

    def process_gate_weight(self, out, index):
        x_s = out[0: int(out.shape[0]//2)]
        x_t = out[out.shape[0]//2: out.shape[0]]
        x_all = torch.cat((x_s, x_t), 2)
        x_all = x_all.view(x_all.shape[0], -1)
        weight = torch.sigmoid(self.bn_lst[index](
            self.gate[index](x_all.float())))
        weight = torch.mean(weight, dim=0)
        res = self.softmax(weight).squeeze()
        return res

    def get_features(self, output_list):
        fea_list_src, fea_list_tar = [], []
        for fea in output_list:
            fea_list_src.append(fea[0: fea.size(0) // 2])
            fea_list_tar.append(fea[fea.size(0) // 2:])
        return fea_list_src, fea_list_tar

    # For Boosting-based
    def forward_Boosting(self, x, weight_mat=None):
        out = self.gru_features(x)
        fea = out[0]
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()

        out_list_all = out[1]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).to(self.device)
        if weight_mat is None:
            weight = (1.0 / self.len_seq *
                      torch.ones(self.num_layers, self.len_seq)).to(self.device)
        else:
            weight = weight_mat
        dist_mat = torch.zeros(self.num_layers, self.len_seq).to(self.device)
        for i in range(len(out_list_s)):
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2], device=self.device)
            for j in range(self.len_seq):
                loss_trans = criterion_transder.compute(
                    out_list_s[i][:, j, :], out_list_t[i][:, j, :])
                loss_transfer = loss_transfer + weight[i, j] * loss_trans
                dist_mat[i, j] = loss_trans
        return fc_out, loss_transfer, dist_mat, weight

    # For Boosting-based
    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
        epsilon = 1e-12
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        ind = dist_new > dist_old + epsilon
        weight_mat[ind] = weight_mat[ind] * \
            (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, self.len_seq)
        return weight_mat

    def predict(self, x):
        out = self.gru_features(x, predict=True)
        fea = out[0]
        if self.use_bottleneck == True:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()
        return fc_out
