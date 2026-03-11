import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import copy
import concurrent.futures as cf
from sklearn.ensemble import RandomForestClassifier
from utils import *

def train_client(client_data, global_model, local_epochs=5, device="cuda"):
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

def test_client(client, client_data, global_models, fusion_module, modalities, excluded_clients=['S06', 'S07', 'S08', 'S09'], device="cuda"):
    fusion_input = []
    modality_accuracies = []
    valid_modalities = modalities.copy()
    if client in excluded_clients:
        valid_modalities = [m for m in modalities if m not in ["Tactile-Left", "Tactile-Right"]]
    for modality in valid_modalities:
        mod_idx = modalities.index(modality)
        with torch.no_grad():
            data, target = client_data[modality]
            data = torch.tensor(np.array(data)).float().to(device)
            target = torch.tensor(target).long().to(device)
            global_model = global_models[mod_idx].to(device)
            output = global_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            modality_accuracies.append(100. * correct / target.size(0))
            fusion_input.append(pred.cpu().numpy())
    fusion_input = np.hstack(fusion_input)
    fusion_accuracy = fusion_module.score(fusion_input, target.cpu().numpy()) * 100
    full_accuracies = []
    for modality in modalities:
        if modality in valid_modalities:
            full_accuracies.append(modality_accuracies.pop(0))
        else:
            full_accuracies.append(float('nan'))
    return full_accuracies + [fusion_accuracy]

def calculate_shapley_values_via_fusion(client, client_data, client_models, modalities, excluded_clients=['S06', 'S07', 'S08', 'S09'], device="cuda"):
    valid_modalities = modalities.copy()
    if client in excluded_clients:
        valid_modalities = [m for m in modalities if m not in ["Tactile-Left", "Tactile-Right"]]
    _, target = client_data[valid_modalities[0]]
    fusion_input = []
    for mod_idx, modality in enumerate(valid_modalities):
        with torch.no_grad():
            data, _ = client_data[modality]
            data = torch.tensor(np.array(data)).float().to(device)
            model = client_models[mod_idx].to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            fusion_input.append(pred.cpu().numpy())

    fusion_input = np.hstack(fusion_input)
    fusion_module = RandomForestClassifier(n_estimators=10).fit(fusion_input, target)

    fusion_input_sampled = shap.sample(fusion_input, 50)
    explainer = shap.TreeExplainer(fusion_module, fusion_input_sampled)
    shap_value = explainer.shap_values(fusion_input_sampled)
    shap_value = np.sum(np.abs(shap_value), axis=(0, 2))
    shap_value = get_aligned_shap_values(shap_value, valid_modalities, modalities)   
    return shap_value

def train_local_fusion_module(client, client_data, trained_global_models, modalities, excluded_clients=['S06', 'S07', 'S08', 'S09'], device="cuda"):
    valid_modalities = modalities.copy()
    if client in excluded_clients:
        valid_modalities = [m for m in modalities if m not in ["Tactile-Left", "Tactile-Right"]]
    _, target = client_data[valid_modalities[0]]
    fusion_input = []
    for modality in valid_modalities:
        with torch.no_grad():
            mod_idx = modalities.index(modality)
            data, _ = client_data[modality]
            data = torch.tensor(np.array(data)).float().to(device)
            model = trained_global_models[mod_idx].to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            fusion_input.append(pred.cpu().numpy())

    fusion_input = np.hstack(fusion_input)
    fusion_module = RandomForestClassifier(n_estimators=10).fit(fusion_input, target)
    accuracy = fusion_module.score(fusion_input, target) * 100
    return fusion_module, accuracy

def compute_modality_priority(shap_value, model_size, recency, modality_weights, top_shap):
    Priority = modality_weights[0]*shap_value + modality_weights[1]*(1-model_size) + modality_weights[2]*recency
    top_indices = np.argsort(Priority, axis=1)[:, :-top_shap]
    Priority[np.arange(Priority.shape[0])[:, None], top_indices] = -1
    return Priority

def compute_client_priority(ite, client_losses_matrix, client_last_selected_round, client_weights, prefer_higher_loss=True):
    client_loss_mean = np.nanmean(client_losses_matrix, axis=1)
    if np.all(np.isnan(client_loss_mean)):
        client_loss_mean = np.zeros_like(client_loss_mean)
    else:
        maxv = np.nanmax(client_loss_mean)
        client_loss_mean[np.isnan(client_loss_mean)] = maxv
    norm_loss = normalize_1d(client_loss_mean)
    staleness = (ite - client_last_selected_round) / (ite + 1)
    norm_staleness = normalize_1d(staleness)
    if prefer_higher_loss:
        loss_term = norm_loss
    else:
        loss_term = 1 - norm_loss
    client_priority = client_weights[0]*loss_term + client_weights[1]*norm_staleness
    return client_priority

def run_local_training_round(client_list, global_models, client_data_train, modalities, local_epochs):
    global_models_copy = [copy.deepcopy(model) for model in global_models]
    client_models = [[] for _ in modalities]
    client_idx_per_mod = [[] for _ in modalities]
    client_loss = [[] for _ in modalities]
    client_par = [[] for _ in modalities]
    shap_values = []
    
    for cidx, client in enumerate(client_list):
        local_models = []
        futs = []
        with cf.ThreadPoolExecutor(max_workers=len(modalities)) as ex:
            for mod_idx, modality in enumerate(modalities):
                if client in ['S06', 'S07', 'S08', 'S09'] and modality in ["Tactile-Left", "Tactile-Right"]:
                    client_par[mod_idx].append(593428)
                    client_loss[mod_idx].append(np.nan)
                    continue
                device = "cuda" if torch.cuda.is_available() else "cpu"
                futs.append(ex.submit(
                    lambda mi, md, dv: train_client(client_data_train[client][md], global_models_copy[mi], local_epochs, device=dv),
                    mod_idx, modality, device
                ))
            idx = 0
            for mod_idx, modality in enumerate(modalities):
                if client in ['S06', 'S07', 'S08', 'S09'] and modality in ["Tactile-Left", "Tactile-Right"]:
                    continue
                client_model, avg_loss = futs[idx].result()
                idx += 1
                local_models.append(client_model)
                client_models[mod_idx].append(client_model.state_dict())
                client_idx_per_mod[mod_idx].append(cidx)
                client_par[mod_idx].append(count_parameters(client_model))
                client_loss[mod_idx].append(avg_loss)
                
        device = "cuda" if torch.cuda.is_available() else "cpu"
        shap_value = calculate_shapley_values_via_fusion(client, client_data_train[client], local_models, modalities, device=device)
        shap_values.append(shap_value)
        
    shap_values = normalization(np.array(shap_values))
    model_size = normalization(np.array(client_par).T)
    return client_models, client_idx_per_mod, client_loss, shap_values, model_size

def select_and_aggregate(args, ite, modalities, shap_vals, model_size, recency_history, client_loss, client_last_selected_round, client_models, client_idx_per_mod, global_models, active_indices):
    modality_recency = (ite - recency_history) / (ite + 1)
    modality_recency = modality_recency[active_indices, :]
    if args.random_modality:
        shap_vals = np.random.rand(*shap_vals.shape)
    Priority = compute_modality_priority(shap_vals, model_size, modality_recency, args.modality_weights, args.top_shap)
    rows, cols = np.where(Priority != -1)
    recency_history[np.array(active_indices)[rows], cols] = ite
    client_losses_matrix = np.array(client_loss).T
    clsr_active = client_last_selected_round[active_indices]
    
    client_priority = compute_client_priority(ite, client_losses_matrix, clsr_active, args.client_weights)
    total_clients = recency_history.shape[0] 
    k = max(1, round(args.client_select_ratio * total_clients))
    k = min(k, len(active_indices))

    if args.random_clients:
        selected_active = np.random.choice(np.arange(len(active_indices)), size=k, replace=False).tolist()
    else:
        selected_active = np.argsort(-client_priority)[:k]
    selected_clients = [active_indices[i] for i in selected_active]
    client_last_selected_round[selected_clients] = ite

    counts = []
    per_mod_selected = []
    for midx, mod in enumerate(modalities):
        allowed = set(np.where(Priority.T[midx] != -1)[0].tolist())
        picked_clients = [int(i) for i in selected_active if i in allowed]
        per_mod_selected.append(picked_clients)
        picked_clients_global = [active_indices[i] for i in picked_clients]
        mapping = {c_idx: w_idx for w_idx, c_idx in enumerate(client_idx_per_mod[midx])}
        chosen_weight_indices = [mapping[c] for c in picked_clients_global if c in mapping]
        gw, cnt = average_weights(client_models[midx], chosen_weight_indices, global_models[midx])
        global_models[midx].load_state_dict(gw)
        counts.append(cnt)
    return np.array(counts)

def evaluate_round(ite, client_list, client_data_train, client_data_test, global_models, modalities, device):
    fusion_modules = {}
    with cf.ThreadPoolExecutor(max_workers=len(client_list)) as ex:
        futs = {client: ex.submit(train_local_fusion_module, client, client_data_train[client], global_models, modalities, device=device) for client in client_list}
        for client in client_list:
            em, _ = futs[client].result()
            fusion_modules[client] = em
            
    with cf.ThreadPoolExecutor(max_workers=len(client_list)) as ex:
        futs2 = {client: ex.submit(test_client, client, client_data_test[client], global_models, fusion_modules[client], modalities, device=device) for client in client_list}
        accs = [futs2[client].result() for client in client_list]
        
    ca = np.array(accs)
    print(f"Iteration {ite} - Fusion mean accuracy: {np.mean(ca[:, -1]):.2f}%")
    return accs

def federated_learning(args, client_data_train, client_data_test, global_models, device):
    modalities = ['Eye','EMG-Left','EMG-Right','Tactile-Left','Tactile-Right','IMU']
    num_clients = len(client_data_train)
    recency_history = np.full((num_clients, len(modalities)), -1)
    client_last_selected_round = np.full((num_clients,), -1)
    accuracy_matrix, modality_counts = [], []
    
    for ite in range(args.iterations):
        client_list = list(client_data_train.keys())[:num_clients]
        active_indices = [client_list.index(c) for c in client_list]
            
        client_models, client_idx_per_mod, client_loss, shap_values, model_size = run_local_training_round(
            client_list, global_models, client_data_train, modalities, args.local_epochs
        )
        counts = select_and_aggregate(
            args, ite, modalities, shap_values, model_size, recency_history, 
            client_loss, client_last_selected_round, client_models, client_idx_per_mod, 
            global_models, active_indices
        )
        modality_counts.append(counts)
        accs = evaluate_round(ite, client_list, client_data_train, client_data_test, global_models, modalities, device)
        accuracy_matrix.append(accs)
        
    return np.array(accuracy_matrix), np.array(modality_counts)
