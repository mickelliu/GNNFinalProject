import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss

from gae.hessian.hessian import *


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight, b_vae=1.0):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + b_vae * KLD


def loss_dc(z_real, z):
    dc_loss_real = nn.BCEWithLogitsLoss()(z_real, torch.ones(z_real.shape))
    dc_loss_fake = nn.BCEWithLogitsLoss()(z, torch.zeros(z.shape))
    return dc_loss_real + dc_loss_fake
    # todo


def loss_gen(z):
    gen_loss = nn.BCEWithLogitsLoss()(z, torch.zeros(z.shape))
    return gen_loss


def loss_hessian(G, z, **G_kwargs):
    return hessian_penalty(G, z, **G_kwargs)
