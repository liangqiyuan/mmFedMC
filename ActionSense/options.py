import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # Environment
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    # Data Partitioning
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--class_non_iid_rate', type=float, default=1.0)
    
    # Federated Learning Settings
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--local_epochs', type=int, default=5)
    
    # Modality Selection
    parser.add_argument('--top_shap', type=int, default=1, help='Number of top modalities to drop')
    parser.add_argument('--modality_weights', nargs='+', type=float, default=[1/3, 1/3, 1/3], 
                        help='Weights for SHAP, model_size, and recency in modality selection')
    parser.add_argument('--random_modality', action='store_true', help='Use random modality selection instead of Priority')

    # Client Selection
    parser.add_argument('--client_select_ratio', type=float, default=0.2, help='Ratio of clients to select per round')
    parser.add_argument('--client_weights', nargs='+', type=float, default=[1.0, 0.0], 
                        help='Weights for loss and staleness in client selection')
    parser.add_argument('--random_clients', action='store_true', help='Use random client selection')
    parser.add_argument('--prefer_higher_loss', action='store_true', default=True, help='Prefer selecting clients with higher loss')

    args = parser.parse_args()
    return args
