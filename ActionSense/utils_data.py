import h5py
import numpy as np
from collections import defaultdict
                    
import torch.nn as nn

def load_and_restructure_hdf5_data(filepath):
    with h5py.File(filepath, 'r') as hdf_file:
        key_map = {
            ('eye-tracking-gaze', 'position'): 'Eye',
            ('myo-left', 'emg'): 'EMG-Left',
            ('myo-right', 'emg'): 'EMG-Right',
            ('tactile-glove-left', 'tactile', 'data'): 'Tactile-Left',
            ('tactile-glove-right', 'tactile', 'data'): 'Tactile-Right',
            ('xsens-joints', 'rotation', 'xzy', 'deg'): 'IMU'}

        example_labels = hdf_file['example_labels'][:]
        example_label_indexes = hdf_file['example_label_indexes'][:]
        example_matrices = {}
        for key in hdf_file.keys():
            if key.startswith("example_matrices_"):
                device_stream = tuple(key.split("_")[2:])
                mapped_key = key_map.get(device_stream, device_stream) 
                example_matrices[mapped_key] = hdf_file[key][:]
        example_subject_ids = hdf_file['example_subject_ids'][:]
        client_data = {}
        unique_client_ids = set(example_subject_ids)
        for client_id in unique_client_ids:
            client_id_str = client_id.decode('utf-8') if isinstance(client_id, bytes) else client_id
            client_data[client_id_str] = {}
            client_indexes = [i for i, id_ in enumerate(example_subject_ids) if id_ == client_id]
            client_labels = [example_label_indexes[i]-1 for i in client_indexes]
            for mapped_key in example_matrices:
                client_datasets = [example_matrices[mapped_key][i] for i in client_indexes]
                client_data[client_id_str][mapped_key] = [client_datasets, client_labels]

        for client_id in ['S06', 'S07', 'S08', 'S09']:
            if client_id in client_data:
                for tactile_key in ['Tactile-Left', 'Tactile-Right']:
                    if tactile_key in client_data[client_id]:
                        for dataset in client_data[client_id][tactile_key][0]:
                            dataset[:] = 0
    return client_data

def dirichlet_partition_data(client_data, alpha, group2_ids=['S06', 'S07', 'S08', 'S09'], seed=42):
    np.random.seed(seed)
    clients = list(client_data.keys())
    group1_clients = [c for c in clients if c not in group2_ids]; group2_clients = [c for c in clients if c in group2_ids]
    group1_data_by_label = defaultdict(list); group2_data_by_label = defaultdict(list)
    
    for client_id, streams in client_data.items():
        ref_key = list(streams.keys())[0]
        ref_datasets, ref_labels = streams[ref_key]
        target_dict = group1_data_by_label if client_id in group1_clients else group2_data_by_label
        for i, label in enumerate(ref_labels):
            sample_data = {mapped_key: (datasets[i], labels[i]) for mapped_key, (datasets, labels) in streams.items() if i < len(datasets)}
            target_dict[label].append((client_id, sample_data))
    
    new_client_data = {client_id: {key: ([], []) for key in client_data[client_id].keys()} for client_id in clients}
    
    for group_clients, group_data in [(group1_clients, group1_data_by_label), (group2_clients, group2_data_by_label)]:
        num_clients = len(group_clients)
        for label, label_data in group_data.items():
            if not label_data:
                continue
            
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            client_sample_counts = np.round(proportions * len(label_data)).astype(int)
            diff = len(label_data) - np.sum(client_sample_counts)
            if diff > 0:
                indices = np.random.choice(num_clients, int(diff), replace=False)
                for index in indices:
                    client_sample_counts[index] += 1
            elif diff < 0:
                indices = np.random.choice(num_clients, int(-diff), replace=False)
                for index in indices:
                    if client_sample_counts[index] > 0:
                        client_sample_counts[index] -= 1
            
            np.random.shuffle(label_data)
            start_idx = 0
            for i, client_id in enumerate(group_clients):
                count = client_sample_counts[i]
                if count > 0:
                    end_idx = min(start_idx + count, len(label_data))
                    for j in range(start_idx, end_idx):
                        for mapped_key, (dataset, data_label) in label_data[j][1].items():
                            if mapped_key in new_client_data[client_id]:
                                new_client_data[client_id][mapped_key][0].append(dataset)
                                new_client_data[client_id][mapped_key][1].append(data_label)
                    start_idx = end_idx
    return new_client_data


def split_client(client_data, train_ratio):
    client_data_train = {}
    client_data_test = {}
    for client, modalities_data in client_data.items():
        client_data_train[client] = {}
        client_data_test[client] = {}
        ref_modality = list(modalities_data.keys())[0]
        _, y = modalities_data[ref_modality]
        unique_classes, class_indices, class_counts = np.unique(y, return_index=True, return_counts=True)
        train_indices = []
        test_indices = []
        for cls, idx, count in zip(unique_classes, class_indices, class_counts):
            all_indices = np.arange(idx, idx + count)
            np.random.shuffle(all_indices)
            boundary = int(count * train_ratio)
            train_indices.extend(all_indices[:boundary])
            test_indices.extend(all_indices[boundary:])
        for device_stream, data in modalities_data.items():
            x = data[0]
            x_train = [x[i] for i in train_indices]
            y_train = [y[i] for i in train_indices]
            x_test = [x[i] for i in test_indices]
            y_test = [y[i] for i in test_indices]
            client_data_train[client][device_stream] = (x_train, y_train)
            client_data_test[client][device_stream] = (x_test, y_test)
    return client_data_train, client_data_test


def split_client_missing_modality(client_data, train_ratio, missing_modality_rate):
    client_data_train = {}
    client_data_test = {}
    for client, modalities_data in client_data.items():
        num_modalities = len(modalities_data)
        num_to_remove = int(num_modalities * missing_modality_rate)
        modalities_to_keep = list(modalities_data.keys())
        np.random.shuffle(modalities_to_keep)
        modalities_to_keep = modalities_to_keep[:max(2, num_modalities - num_to_remove)]
        client_data_train[client] = {}
        client_data_test[client] = {}
        ref_modality = modalities_to_keep[0]
        _, y = modalities_data[ref_modality]
        unique_classes, class_indices, class_counts = np.unique(y, return_index=True, return_counts=True)
        train_indices = []
        test_indices = []
        for cls, idx, count in zip(unique_classes, class_indices, class_counts):
            all_indices = np.arange(idx, idx + count)
            np.random.shuffle(all_indices)
            boundary = int(count * train_ratio)
            train_indices.extend(all_indices[:boundary])
            test_indices.extend(all_indices[boundary:])
        for device_stream in modalities_to_keep:
            data = modalities_data[device_stream]
            x = data[0]
            x_train = [x[i] for i in train_indices]; y_train = [y[i] for i in train_indices]
            x_test = [x[i] for i in test_indices]; y_test = [y[i] for i in test_indices]
            client_data_train[client][device_stream] = (x_train, y_train)
            client_data_test[client][device_stream] = (x_test, y_test)
    return client_data_train, client_data_test



class Eye_LSTM(nn.Module):
    def __init__(self):
        super(Eye_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 20)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return nn.LogSoftmax(dim=1)(x)

class EMG_LSTM(nn.Module):
    def __init__(self):
        super(EMG_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 20)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return nn.LogSoftmax(dim=1)(x)

class Tactile_LSTM(nn.Module):
    def __init__(self):
        super(Tactile_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=32*32, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 20)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return nn.LogSoftmax(dim=1)(x)

class IMU_LSTM(nn.Module):
    def __init__(self):
        super(IMU_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=22 * 3, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 20)
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return nn.LogSoftmax(dim=1)(x)