# GNNFinalProject
For 《图神经网络》课程 Final Project

### Adapted from: 
https://github.com/wpeebles/hessian_penalty  
https://github.com/zfjsail/gae-pytorch

### Tips and Instructions
Straight running from the terminal will likely not work, so I suggest you running train.py in an interactive environment like Python Console

In main.py under main function, comment out the experiment that you don't want to run, and leave only the experiment you intend to run uncommented.

### Experiment #1
To check if Hessian does a better job in disentanglement than vanilla GAE by looking at the correlation plots

### Experiment #2
To check if Hessian does increase the accurarcy of model's predictions

### Experiment #3
To check if b-GAE does a good job in disentanglement

### Conclusion
Hessian does not work for simple GCN in link-prediciton task. Might because of its lack of complexity in model architecture.
