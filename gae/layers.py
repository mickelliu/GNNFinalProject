import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn as nn


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class DisConv(Module):
    def __init__(self, in_dim, channels, C_dim, iterations=3, beta=1):  # 输入维数， 通道数目， 每个通道的输出维数， 迭代次数， 平衡因子
        super(DisConv, self).__init__()
        self.channels = channels
        self.in_dim = in_dim
        self.c_dim = C_dim
        self.iterations = iterations
        self.beta = beta
        self.weight_list = []
        self.bias_list = []
        self.relu = nn.ReLU()
        self.weight_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(self.in_dim, self.c_dim), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))
        self.bias_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(1, self.c_dim), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))
        self.init_parameters()

    def init_parameters(self):
        for i, item in enumerate(self.parameters()):
            torch.nn.init.normal_(item, mean=0, std=1)

    def forward(self, features, adj):
        c_features = []
        for i in range(self.channels):
            z = torch.mm(features, self.weight_list[i]) + self.bias_list[i]
            z = F.normalize(z, dim=1)
            c_features.append(z)
        out_features = c_features
        for l in range(self.iterations):
            c_attentions = []
            for i in range(self.channels):
                channel_f = out_features[i]
                c_attentions.append(self.parse_attention(adj, channel_f))
            all_attentions = torch.cat([c_attentions[i] for i in range(len(c_attentions))], dim=2)
            all_attentions = F.softmax(all_attentions, dim=2)
            neg_all_attention = torch.zeros_like(all_attentions)
            adj_all = torch.unsqueeze(adj.to_dense(), dim=2).repeat(1, 1, self.channels)
            all_attentions = torch.where(adj_all > 0, all_attentions, neg_all_attention)
            for k in range(self.channels):
                feat = out_features[k]
                atte = all_attentions[:, :, k].squeeze()
                out_features[k] = (F.normalize(feat + torch.mm(atte, feat), dim=1))
        output = torch.cat([out_features[i] for i in range(len(out_features))], dim=1)
        return output

    def parse_attention(self, adj, features):
        attention_matrix = torch.mm(features, features.t())
        neg_attention = torch.zeros_like(attention_matrix)
        attention_matrix = torch.where(adj.to_dense() > 0, attention_matrix, neg_attention)
        attention_matrix = attention_matrix * 1.0 / (self.beta)
        attention_matrix = torch.unsqueeze(attention_matrix, dim=2)

        return attention_matrix