import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default = 'cuda:0')
    parser.add_argument('--train_ratio', type=float, default = 0.8)
    parser.add_argument('--class_non_IID_rate', type=float, default = 1.0)
    parser.add_argument('--modality_non_IID_rate', type=float, default = 0.0)

    parser.add_argument('--gamma', type=int, default = 1)
    parser.add_argument('--delta', type=float, default = 0.2)
    parser.add_argument('--alpha_shapley', type=float, default = 1/3)
    parser.add_argument('--alpha_comm', type=float, default = 1/3)
    parser.add_argument('--alpha_recency', type=float, default = 1/3)

    parser.add_argument('--round', type=int, default = 100)
    parser.add_argument('--epoch', type=int, default = 5)

    args = parser.parse_args()
    return args