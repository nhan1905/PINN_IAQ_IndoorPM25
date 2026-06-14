import os
import gc
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def clear_GPU_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    print("GPU memory has been cleared!")

def batch_size_optimization(window_size, feature_dim, safety_factor=0.7):
    if not torch.cuda.is_available():
        return 32 
    
    total_vram = torch.cuda.get_device_properties(0).total_memory
    sample_size = window_size * feature_dim * 4 * 10 
    suggested_batch = int((total_vram * safety_factor) / sample_size)
    
    CUDA_batch_sizes = [32, 64, 128, 256, 512]
    optimized_batch_size = max([b for b in CUDA_batch_sizes if b <= suggested_batch])
    
    return optimized_batch_size

def create_3d_tensor(
                        data_feature,
                        data_target,
                        input_size,
                        window_size,
                        pred_steps=1,
                        num_workers=4,
                        mode='training' # 'training' or 'evaluation'
                    ):

    batch_size = batch_size_optimization(window_size, input_size)
    
    stride = window_size + pred_steps if (mode == 'evaluation') else 1
    shuffle = False if mode == 'evaluation' else True

    X, Y, Feat_seq = [], [], []

    for i in range(0, len(data_feature) - window_size - pred_steps + 1, stride):

        pt_feature_window = data_feature[i : i + window_size]
        pt_target_window = data_target[i : i + window_size].reshape(-1, 1)
        pt_window_data = np.concatenate((pt_feature_window, pt_target_window), axis=1)
        X.append(pt_window_data)

        target_sequence = data_target[i + window_size : i + window_size + pred_steps]
        Y.append(target_sequence)

        future_features = data_feature[i + window_size : i + window_size + pred_steps]
        Feat_seq.append(future_features)
        
    X_3D = np.array(X, dtype=np.float32)
    Y_3D = np.array(Y, dtype=np.float32)
    if Y_3D.ndim == 3:
        Y_3D = Y_3D.squeeze(-1)  # [N, pred_steps, 1] → [N, pred_steps]
    Feat_3D = np.array(Feat_seq, dtype=np.float32)

    X_3D_tensor = torch.from_numpy(X_3D)
    Y_3D_tensor = torch.from_numpy(Y_3D).unsqueeze(-1)  # [N, pred_steps, 1]
    Feat_3D_tensor = torch.from_numpy(Feat_3D)

    print(f"--- Created 3D Tensor ---")
    print(f"X: {X_3D_tensor.shape} | Y_seq: {Y_3D_tensor.shape} | Future_Feat: {Feat_3D_tensor.shape}")

    dataset = TensorDataset(X_3D_tensor, Y_3D_tensor, Feat_3D_tensor)

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=True
                            )

    return dataloader

def load_model(model, regressor=None, device=None, model_name=None, save_path=None):

    if save_path is not None and model_name is not None:
        save_model_path = os.path.join(save_path, model_name)
    else:
        raise ValueError("save_path and model_name must be provided")

    checkpoint = torch.load(save_model_path, map_location=device, weights_only=True)

    encoder_key = 'trained_encoder_state_dict'
    regressor_key = 'trained_regressor_state_dict'

    model.encoder.load_state_dict(checkpoint[encoder_key])
    if regressor is not None and regressor_key in checkpoint:
        regressor.load_state_dict(checkpoint[regressor_key])

    print(f"--- Successfully loaded model from: {save_model_path} ---")

    model.to(device)
    if regressor is not None:
        regressor.to(device)

    return model, regressor