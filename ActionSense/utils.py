import numpy as np
import torch

def average_weights(weights_list, chosen_weight_indices, global_model):
    model_weights = [weights_list[idx] for idx in chosen_weight_indices if 0 <= idx < len(weights_list)]
    if not model_weights:
        return global_model.state_dict(), 0
    device = next(global_model.parameters()).device
    avg_weights = {}
    for key in model_weights[0].keys():
        tensors = [weights[key].to(device) for weights in model_weights]
        avg_weights[key] = torch.mean(torch.stack(tensors), dim=0)
    return avg_weights, len(model_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normalization(arr):
    min_vals = np.min(arr, axis=1, keepdims=True)
    max_vals = np.max(arr, axis=1, keepdims=True)
    scaled_arr = (arr - min_vals) / (max_vals - min_vals + 1e-10)
    return scaled_arr

def normalize_1d(arr):
    x = np.array(arr, dtype=float)
    min_v = np.nanmin(x)
    max_v = np.nanmax(x)
    return (x - min_v) / (max_v - min_v + 1e-10)

def get_aligned_shap_values(shap_values, valid_modalities, all_modalities):
    aligned_shap_values = np.zeros(len(all_modalities))
    for idx, modality in enumerate(all_modalities):
        if modality in valid_modalities:
            mod_idx = valid_modalities.index(modality)
            aligned_shap_values[idx] = shap_values[mod_idx]
    return aligned_shap_values
