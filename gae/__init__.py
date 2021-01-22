import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=69, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=4, help='Number of units in hidden layer 2.')

# parser.add_argument('--hidden3', type=int, default=64, help='Number of units in hidden layer 3.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='citeseer', help='type of dataset.')
parser.add_argument('--enable_hessian', action="store_true", default=True, help='Hessian Penalty Weight')
parser.add_argument('--lambda_H', type=float, default=0.1, help='Hessian Penalty Weight')
parser.add_argument('--b_vae', type=float, default=1.0, help='Beta')

args = parser.parse_args()
