from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from tqdm import tqdm

from gae.model import GCNModelVAE, Discriminator
from gae.optimizer import loss_function, loss_dc, loss_gen
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=32, help='Number of units in hidden layer 2.')
parser.add_argument('--hidden3', type=int, default=64, help='Number of units in hidden layer 3.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')

args = parser.parse_args()


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    # D = Discriminator(args.hidden1, args.hidden2, args.hidden3)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer_dc = optim.Adam(D.parameters(), lr=args.lr)

    hidden_emb = None

    z_real = torch.tensor(np.random.randn(adj.shape[0], args.hidden2)).float()

    with tqdm(total=args.epochs, postfix=dict, mininterval=0.3) as pbar:
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            # D.train()
            model.zero_grad()
            # D.zero_grad()
            recovered, mu, logvar, z = model(features, adj_norm)
            loss = loss_function(preds=recovered, labels=adj_label,
                                 mu=mu, logvar=logvar, n_nodes=n_nodes,
                                 norm=norm, pos_weight=pos_weight)
            loss.backward(retain_graph=True)
            cur_loss = loss.item()

            # D_z = D(z)
            # D_z_real = D(z_real)
            # dc_loss = loss_dc(D_z_real, D_z)
            # gen_loss = loss_gen(D_z)
            # dc_loss.backward(retain_graph=True)
            # gen_loss.backward()

            optimizer.step()
            # optimizer_dc.step()

            hidden_emb = mu.data.numpy()
            roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

            pbar.set_postfix(**{'train_loss': "{:.5f}".format(cur_loss),
                                # 'dc_loss': "{:.5f}".format(dc_loss),
                                # 'gen_loss': "{:.5f}".format(gen_loss),
                                "val_ap=": "{:.5f}".format(ap_curr),
                                "time=": "{:.5f}".format(time.time() - t), })
            pbar.update(1)

    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    gae_for(args)
