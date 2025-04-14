import shap
import copy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
                    
import torch
import torch.nn as nn
import torch.optim as optim

from options import args_parser
args = args_parser()
device = args.device

def average_weights(weights_list, Priority, global_model, client_loss, select_rate):
    valid_indices = [idx for idx in range(len(weights_list)) if weights_list[idx] != []]
    valid_weights_list = [weights_list[idx] for idx in valid_indices]
    valid_client_losses = [client_loss[idx] for idx in valid_indices]
    sorted_clients_indices = sorted(enumerate(valid_client_losses), key=lambda x: x[1], reverse=False)
    num_selected_clients = round(select_rate * len(valid_weights_list))
    selected_valid_indices = [valid_indices[idx] for idx, _ in sorted_clients_indices[:num_selected_clients]]
    weights_list = [weights_list[idx] for idx in selected_valid_indices]
    Priority = np.where(Priority != -1)[0]
    Priority = [idx for idx in Priority if idx < len(weights_list)]
    model_weights = [weights_list[idx] for idx in Priority]
    
    if not model_weights:
        return global_model.state_dict(), 0
    avg_weights = {}
    for key in model_weights[0].keys():
        avg_weights[key] = torch.mean(torch.stack([weights[key] for weights in model_weights]), dim=0)  
    return avg_weights, len(model_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normalization(arr):
    min_vals = np.nanmin(arr, axis=1, keepdims=True)
    max_vals = np.nanmax(arr, axis=1, keepdims=True)
    scaled_arr = (arr - min_vals) / (max_vals - min_vals + 1e-10)
    return scaled_arr

def get_aligned_shap_values(shap_values, valid_modalities, all_modalities):
    aligned_shap_values = np.zeros(len(all_modalities))
    for idx, modality in enumerate(all_modalities):
        if modality in valid_modalities:
            mod_idx = valid_modalities.index(modality)
            aligned_shap_values[idx] = shap_values[mod_idx]
    return aligned_shap_values
    
def train_client(client_data, global_model, local_epochs):
    criterion = nn.NLLLoss()
    client_model = type(global_model)().to(device)
    client_model.load_state_dict(copy.deepcopy(global_model).state_dict())
    optimizer = optim.SGD(client_model.parameters(), lr=0.1)
    data, target = client_data
    data = torch.tensor(np.array(data)).float().to(device)
    target = torch.tensor(target).long().to(device)
    data_loader = torch.utils.data.DataLoader(list(zip(data, target)), batch_size=32, shuffle=True)

    for epoch in range(local_epochs):
        total_loss = 0
        for batch_data, batch_target in data_loader:
            optimizer.zero_grad()
            output = client_model(batch_data)
            loss = criterion(output, batch_target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(data_loader)
    return client_model, avg_loss

def test_client(client_data, global_models, ensemble_model, modalities):
    ensemble_input = []
    modality_accuracies = []
    valid_modalities = list(client_data.keys())
    for mod_idx, modality in enumerate(modalities):
        if modality in valid_modalities:
            with torch.no_grad():
                data, target = client_data[modality]
                data = torch.tensor(np.array(data)).float().to(device)
                target = torch.tensor(target).long().to(device)
                global_model = global_models[mod_idx].to(device)
                output = global_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                modality_accuracies.append(100. * correct / target.size(0))
                ensemble_input.append(pred.cpu().numpy())
    ensemble_input = np.hstack(ensemble_input)
    ensemble_accuracy = ensemble_model.score(ensemble_input, target.cpu().numpy()) * 100
    full_accuracies = []
    for modality in modalities:
        if modality in valid_modalities:
            full_accuracies.append(modality_accuracies.pop(0))
        else:
            full_accuracies.append(float('nan'))
    return full_accuracies + [ensemble_accuracy]

def ensemble_learning(client_data, client_models, modalities):
    valid_modalities = list(client_data.keys())
    _, target = client_data[valid_modalities[0]]
    ensemble_input = []
    expected_classes = np.array(list(range(20)))
    encoder = LabelEncoder()
    encoder.classes_ = expected_classes
    target = encoder.transform(target)
    for mod_idx, modality in enumerate(modalities):
        if modality in valid_modalities:
            with torch.no_grad():
                data, _ = client_data[modality]
                data = torch.tensor(np.array(data)).float().to(device)
                model = client_models[mod_idx].to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                ensemble_input.append(pred.cpu().numpy())

    ensemble_input = np.hstack(ensemble_input)
    ensemble_model = RandomForestClassifier(n_estimators=10).fit(ensemble_input, target)

    ensemble_input = shap.sample(ensemble_input, 50)
    explainer = shap.TreeExplainer(ensemble_model, ensemble_input)
    shap_values = explainer.shap_values(ensemble_input)
    shap_values = np.sum(np.abs(shap_values), axis=(0, 2))
    shap_values = get_aligned_shap_values(shap_values, valid_modalities, modalities)   
    return ensemble_model, shap_values

def ensemble_learning_V2(client_data, trained_global_models, modalities):
    valid_modalities = list(client_data.keys())
    _, target = client_data[valid_modalities[0]]
    ensemble_input = []
    expected_classes = np.array(list(range(20)))
    encoder = LabelEncoder()
    encoder.classes_ = expected_classes
    target = encoder.transform(target)
    for mod_idx, modality in enumerate(modalities):
        if modality in valid_modalities:
            with torch.no_grad():
                data, _ = client_data[modality]
                data = torch.tensor(np.array(data)).float().to(device)
                model = trained_global_models[mod_idx].to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                ensemble_input.append(pred.cpu().numpy())

    ensemble_input = np.hstack(ensemble_input)
    ensemble_model = RandomForestClassifier(n_estimators=10).fit(ensemble_input, target)
    return ensemble_model

def MFedMC(global_models, client_data_train, client_data_test, round, epoch, clients, gamma, config, delta):
    accuracy_matrix = []
    modality_counts = []
    modalities = ['Eye', 'EMG-Left', 'EMG-Right', 'Tactile-Left', 'Tactile-Right', 'IMU']
    modality_sizes = {'Eye': 70164, 'EMG-Left': 73236, 'EMG-Right': 73236, 'Tactile-Left': 593428, 'Tactile-Right': 593428, 'IMU': 102932}
    recency_history = np.full((clients, len(modalities)), -1)

    for ite in range(round):
        global_models_copy = [copy.deepcopy(model) for model in global_models]
        client_models = [[] for _ in modalities]
        ensemble_models = {}
        shap_value = []
        client_model_par_num = [[] for _ in modalities]
        client_loss = [[] for _ in modalities]
        
        for _, client in enumerate(list(client_data_train.keys())[:clients]):
            local_models = []

            for mod_idx, modality in enumerate(modalities):
                if modality not in client_data_train[client].keys():
                    client_model_par_num[mod_idx].append(modality_sizes[modality])
                    client_loss[mod_idx].append(np.nan)
                    local_models.append([])
                    client_models[mod_idx].append([])
                    continue
                client_model, avg_loss = train_client(client_data_train[client][modality], global_models_copy[mod_idx], local_epochs=epoch)
                local_models.append(client_model)
                client_models[mod_idx].append(client_model.state_dict())
                client_model_par_num[mod_idx].append(count_parameters(client_model))
                client_loss[mod_idx].append(avg_loss)

            ensemble_model, shap_values = ensemble_learning(client_data_train[client], local_models, modalities)
            shap_value.append(shap_values)

        shap_value = normalization(np.array(shap_value))
        model_size = normalization(np.array(client_model_par_num).T)
        recency = (ite - recency_history) / (ite + 1)
        Priority = config[1] * shap_value + config[2] * (1-model_size) + config[3] * recency 
        top_indices = np.argsort(Priority, axis=1)[:, :-gamma]
        Priority[np.arange(Priority.shape[0])[:, None], top_indices] = -1
        recency_history[Priority > 0] = ite

        modality_counts_temp = []
        for mod_idx in range(len(modalities)):
            global_weights, count = average_weights(client_models[mod_idx], Priority.T[mod_idx], global_models[mod_idx], client_loss[mod_idx], delta)
            global_models[mod_idx].load_state_dict(global_weights)
            modality_counts_temp.append(count)
        modality_counts.append(np.array(modality_counts_temp))

        for _, client in enumerate(list(client_data_train.keys())[:clients]):
            ensemble_model = ensemble_learning_V2(client_data_train[client], global_models, modalities)
            ensemble_models[client] = ensemble_model
        
        client_accuracies = [test_client(client_data_test[client], global_models, ensemble_models[client], modalities) for client in (list(client_data_test.keys())[:clients])]
        accuracy_matrix.append(client_accuracies)

    return np.array(accuracy_matrix), np.array(modality_counts)
    