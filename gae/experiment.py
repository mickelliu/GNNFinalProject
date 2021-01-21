import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from gae.train import args, gae_for


def experiment_1():
    # Experiment # 1
    # Different hidden dimension

    args.epochs = 200
    args.enable_hessian = True

    z_output = []
    hidden_2_list = [4, 6, 12, 18, 32]

    for hidden_2 in hidden_2_list:
        args.hidden2 = hidden_2
        z_out, _, _ = gae_for(args)
        z_out_pd = pd.DataFrame(z_out.cpu().detach().numpy())
        z_output.append(z_out_pd)

    fig, ax = plt.subplots(nrows=1, ncols=5)
    for i in range(len(ax)):
        if i == len(ax) - 1:
            cbar = True
        else:
            cbar = False
        sns.heatmap(z_output[i].corr().abs(), cmap="rocket", ax=ax[i], cbar=cbar)

    plt.show()


def experiment_2():
    # Experiment # 2
    # To verify performance increase

    args.epochs = 200
    args.enable_hessian = True
    args.hidden2 = 6
    num_trials = 10

    avg_roc, avg_ap = 0, 0

    print(f"Enable Hessian = {args.enable_hessian}, Hidden_2 = {args.hidden2}")
    for i in range(num_trials):
        _, roc_score, ap_score = gae_for(args)
        avg_roc += roc_score
        avg_ap += ap_score

    print(f"Average AP Score = {avg_ap/num_trials}, Average ROC Score = {avg_roc/num_trials}")