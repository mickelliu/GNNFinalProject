from __future__ import division
from __future__ import print_function

import time

import numpy as np
import scipy.sparse as sp
from torch import optim
from tqdm import tqdm

from src.model import *
from src.optimizer import loss_function, loss_hessian
from src.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score


def gae_for(args):
    # GPU
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

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
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    embedder = GCNEmbedder(feat_dim, args.hidden1, args.dropout).to(device)
    encoder = GCNEncoder(args.hidden1, args.hidden2, args.dropout).to(device)

    optimizer_embedding = optim.Adam(embedder.parameters(), lr=args.lr)
    optimizer_encoding = optim.Adam(encoder.parameters(), lr=args.lr)

    hidden_emb = None

    features, adj_norm, pos_weight, adj_label = features.to(device), adj_norm.to(device), pos_weight.to(
        device), adj_label.to(device)

    with tqdm(total=args.epochs, postfix=dict, mininterval=0.3) as pbar:
        for epoch in range(args.epochs):
            t = time.time()
            embedder.zero_grad()
            encoder.zero_grad()

            embedder.train()
            encoder.train()

            z_in = embedder(features, adj_norm)
            recovered, mu, logvar, z_out = encoder(z_in, adj_norm)

            loss = loss_function(preds=recovered, labels=adj_label,
                                 mu=mu, logvar=logvar, n_nodes=n_nodes,
                                 norm=norm, pos_weight=pos_weight, b_vae=args.b_vae)

            if args.enable_hessian:
                hessian_loss = loss_hessian(encoder, z_in, adj=adj_norm)
                loss += args.lambda_H * hessian_loss

            loss.backward(retain_graph=True)

            optimizer_embedding.step()
            optimizer_encoding.step()

            cur_loss = loss.item()
            hidden_emb = mu.cpu().data.numpy()
            roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

            pbar.set_postfix(**{'train_loss': "{:.5f}".format(cur_loss),
                                "val_ap=": "{:.5f}".format(ap_curr),
                                "time=": "{:.5f}".format(time.time() - t), })
            pbar.update(1)

    print(f"Experiment Name: hidden_2 = {args.hidden2} -----> Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    print("")
    return z_out, roc_score, ap_score
