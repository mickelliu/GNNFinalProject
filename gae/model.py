import torch
import torch.nn as nn
import torch.nn.functional as F

from gae.layers import GraphConvolution


class GCNEmbedder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, dropout):
        super(GCNEmbedder, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)

    def embed(self, x, adj):
        return self.gc1(x, adj)

    def forward(self, x, adj):
        return self.embed(x, adj)


class GCNEncoder(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, dropout):
        super(GCNEncoder, self).__init__()
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, z_in, adj):
        return self.gc2(z_in, adj), self.gc3(z_in, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, z_in, adj):
        mu, logvar = self.encode(z_in, adj)
        z_out = self.reparameterize(mu, logvar)
        return self.dc(z_out), mu, logvar, z_out


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
