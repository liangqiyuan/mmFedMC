from utils_data import *
from utils_train import *

from options import args_parser

if __name__ == '__main__':
    args = args_parser()

    filepath = './data/data_processed_allStreams_10s_10hz_5subj_ex20-20_allActs.hdf5'
    client_data = load_and_restructure_hdf5_data(filepath)

    if args.class_non_IID_rate < 1 and args.modality_non_IID_rate == 0:
        print("Class non-IID setting")
        client_data_non_IID = dirichlet_partition_data(client_data, non_IID_ratio=args.class_non_IID_rate)
        client_data_train, client_data_test = split_client(client_data_non_IID, train_ratio=args.train_ratio)
    elif args.modality_non_IID_rate > 0 and args.class_non_IID_rate == 1:
        print("Modality non-IID setting")
        client_data_train, client_data_test = split_client_missing_modality(client_data, train_ratio=args.train_ratio, missing_modality_rate=args.modality_non_IID_rate)
    else:
        raise ValueError("Invalid configuration: Please set either class_non_IID_rate < 1 or modality_non_IID_rate > 0, not both.")


    config = [args.alpha_shapley, args.alpha_comm, args.alpha_recency]
    models = [Eye_LSTM(), EMG_LSTM(), EMG_LSTM(), Tactile_LSTM(), Tactile_LSTM(), IMU_LSTM()]
    accuracy_matrix, modality_counts = MFedMC(models, client_data_train, client_data_test, round=args.round, epoch=args.epoch, clients=len(client_data_train), gamma=args.gamma, config=config, delta=args.delta)
    file_name = f'results/MFedMC_ActionSense_Top_{args.gamma}_ShapCommRec_{config[0]:.1f}_{config[1]:.1f}_{config[2]:.1f}_Client_{config[3]:.1f}.npz'
    np.savez(file_name, acc=accuracy_matrix, mod=modality_counts)

