import statistics

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gae.train import gae_for
from gae import args


def experiment_1(enable_hessian):
    # Experiment # 1
    # Different hidden dimension
    args.epochs = 200
    if enable_hessian:
        args.enable_hessian = True
    else:
        args.enable_hessian = False

    z_output = []
    hidden_2_list = [4, 6, 12, 18, 24, 32, 40]

    print(f"Enable Hessian = {args.enable_hessian}, Hidden_2 = {hidden_2_list}")

    for hidden_2 in hidden_2_list:
        args.hidden2 = hidden_2
        z_out, _, _ = gae_for(args)
        z_out_pd = pd.DataFrame(z_out.cpu().detach().numpy())
        z_output.append(z_out_pd)

    fig, ax = plt.subplots(nrows=1, ncols=len(hidden_2_list))
    fig.suptitle(f"Enable Hessian = {args.enable_hessian}")
    for i in range(len(ax)):
        if i == len(ax) - 1:
            cbar = True
        else:
            cbar = False
        sns.heatmap(z_output[i].corr().abs(), ax=ax[i], cbar=cbar, cmap="YlGn")
        ax[i].set_title(f'k = {hidden_2_list[i]}')

    plt.show()


def experiment_2(enable_hessian, hidden2, lambda_H=0.1):
    # Experiment # 2
    # To verify performance increase

    if enable_hessian:
        args.enable_hessian = True
    else:
        args.enable_hessian = False

    args.epochs = 100
    args.hidden2 = hidden2
    args.lambda_H = lambda_H

    num_trials = 10

    ap, roc = [], []

    print("<<< ===== Experiment #2 ===== >>>")
    print(f"Enable Hessian = {args.enable_hessian}, Hidden_2 = {args.hidden2}, Lambda_H = {args.lambda_H}")
    for i in range(num_trials):
        print(f"Trial #{i}")
        _, roc_score, ap_score = gae_for(args)
        ap.append(ap_score)
        roc.append(roc_score)

    print(
        f"<<< ===== Enable Hessian = {args.enable_hessian}, Hidden_2 = {args.hidden2}, Lambda_H = {args.lambda_H} ===== >>>")
    print(f"Average AP Score = {statistics.mean(ap)}, AP_std = {statistics.stdev(ap)}")
    print(f"Average ROC Score = {statistics.mean(roc)}, ROC_std = {statistics.stdev(roc)}")


def experiment_3(enable_hessian):
    # Experiment # 3
    # b_vae
    args.epochs = 200

    if enable_hessian:
        args.enable_hessian = True
    else:
        args.enable_hessian = False

    z_output = []
    hidden_2_list = [6, 12, 24]
    beta_list = [1, 10, 100, 500, 1000]

    print(f"Enable Hessian = {args.enable_hessian}, Hidden_2 = {hidden_2_list}")

    for row, hidden_2 in enumerate(hidden_2_list):
        z_output.append([])
        for b_vae in beta_list:
            args.hidden2 = hidden_2
            args.b_vae = b_vae
            print(
                f"<<< ===== Enable Hessian = {args.enable_hessian}, Hidden_2 = {args.hidden2}, Lambda_H = {args.lambda_H}, b_vae = {args.b_vae} ===== >>>")
            z_out, _, _ = gae_for(args)
            z_out_pd = pd.DataFrame(z_out.cpu().detach().numpy())
            z_output[row].append(z_out_pd)

    cols = ['beta = {}'.format(col) for col in beta_list]
    rows = ['k = {}'.format(row) for row in hidden_2_list]

    fig, axes = plt.subplots(nrows=len(hidden_2_list), ncols=len(beta_list))
    fig.suptitle(f"B_VAE Disentanglement")

    for j in range(len(hidden_2_list)):
        for i in range(len(beta_list)):
            if i == len(beta_list) - 1:
                cbar = True
            else:
                cbar = False
            sns.heatmap(z_output[j][i].corr().abs(), ax=axes[j][i], cbar=cbar, cmap="YlGn")

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, size='large')

    fig.tight_layout()

    plt.show()

    return z_output
