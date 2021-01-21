import torch
import torch.nn as nn
import torch.nn.functional as F

from gae.layers import GraphConvolution, DisConv

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.dc1 = DisConv(input_feat_dim, 8, 4)
        self.dc2 = DisConv(32, 8, 4)
        self.dc3 = DisConv(32, 8, 4)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        # hidden1 = self.gc1(x, adj)
        # return self.gc2(hidden1, adj), self.gc3(hidden1, adj)
        hidden1 = F.relu(self.dc1(x, adj))
        return self.dc2(hidden1, adj), self.dc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class Discriminator(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Discriminator, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc2 = nn.Linear(hidden_dim3, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, 1)
        # todo

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x).sigmoid_()
        # todo
