import os
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


# A. GENERAL FUNCTION DEFINES
def clear_GPU_memory():
    # Collect garbage of Python (RAM)
    gc.collect()
    
    # Delete unused cache of PyTorch in VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    print("GPU memory has been cleared!")

def batch_size_optimization(window_size, feature_dim, safety_factor=0.7):
    """
    Suggest batch_size based on available VRAM.
    """
    # Default batch size for CPU
    if not torch.cuda.is_available():
        return 32 
    
    # Get available GPUVRAM (in bytes)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    
    # Estimate sample size (float32 = 4 bytes) | Need to store both input and output for Autoencoder, plus parameters/gradients
    sample_size = window_size * feature_dim * 4 * 10 
    
    # Calculate batch_size based on safety factor (avoid out of memory)
    suggested_batch = int((total_vram * safety_factor) / sample_size)
    
    # Usually choose batch_size as a power of 2 (32, 64, 128, 256, 512, 1024) to optimize work distribution in CUDA
    #powers_of_2 = [2**i for i in range(5, 12)] # 32 to 2048
    powers_of_2 = [32, 64, 128, 256, 512]
    optimized_batch_size = max([b for b in powers_of_2 if b <= suggested_batch])
    
    return optimized_batch_size

def create_3d_tensor(
                        data_feature,
                        data_target,
                        input_size,
                        window_size,
                        shuffle = True,
                        num_workers = 4, 
                        mode = 'training' # 'fine-tuning' or 'evaluation'
                    ):
    """
    Transform 2D data into 3D Tensor and package into DataLoader.
    """
    # Create 3D Tensor by Sliding Window
    X, Y = [], [] # Shape [window_size, input_size] and [window_size, 1]
    
    # Create 3D Tensor by Sliding Window
    for i in range(1, len(data_feature) - window_size + 1):
        pt_feature_window = data_feature[i : i + window_size]
        pt_target_window = data_target[i-1 : i + window_size - 1].reshape(-1, 1)

        # Concatenate feature and target
        pt_window_data = np.concatenate((pt_feature_window, pt_target_window), axis=1)
        ft_prediction_target = data_target[i + window_size -1] # Target is the last value of the window

        # Append to X and Y
        X.append(pt_window_data)
        Y.append(ft_prediction_target)
        
    X_3D = np.array(X, dtype=np.float32) # Shape: (Number of samples, Window_size, Input_size)
    Y_3D = np.array(Y, dtype=np.float32) # Shape: (Number of samples, Window_size, 1)

    # If mode is fine-tuning, reshape Y_3D to (Number of samples, 1)
    if mode == ('training' or 'evaluation'):
        Y_3D = Y_3D.reshape(-1, 1)

    # Convert to PyTorch Tensor
    X_3D_tensor = torch.from_numpy(X_3D)
    Y_3D_tensor = torch.from_numpy(Y_3D)

    print(f"--- Created 3D Tensor for the {mode} model with X-shape: {X_3D.shape} and Y-shape: {Y_3D.shape} ---")

    # Create Dataset and DataLoader
    batch_size = batch_size_optimization(window_size, input_size)

    dataset = TensorDataset(X_3D_tensor, Y_3D_tensor)

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

    checkpoint = torch.load(save_model_path, map_location=device)

    # Load trained BiLSTM + Regressor (dict format only)
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
