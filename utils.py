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

def create_3d_tensor_aer_visualization(
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
        Y_3D = Y_3D.squeeze(-1)  
    Feat_3D = np.array(Feat_seq, dtype=np.float32)

    X_3D_tensor = torch.from_numpy(X_3D)
    Y_3D_tensor = torch.from_numpy(Y_3D) # Shape: [batch, prediction_steps, 1] 
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

def create_3d_tensor(
                        data_feature,
                        data_target,
                        input_size,
                        window_size,
                        pred_steps=1,
                        stride=1,
                        num_workers=4,
                        mode="training"
                    ):
    """
    Create time-series samples for multi-step forecasting.

    Output shapes:
        X           : [samples, window_size, input_size]
        Y_seq       : [samples, pred_steps, 1]
        Future_Feat : [samples, pred_steps, feature_dim]

    Parameters
    ----------
    data_feature : np.ndarray
        Input features with shape [time_steps, feature_dim].

    data_target : np.ndarray
        Target ID_PM25 with shape [time_steps] or [time_steps, 1].

    input_size : int
        Number of past input variables, including past target.

    window_size : int
        Number of historical time steps.

    pred_steps : int
        Number of future steps to predict.

    stride : int
        Distance between the starting positions of consecutive samples.
        stride=1 creates overlapping samples.

    num_workers : int
        Number of DataLoader workers.

    mode : str
        "training" or "evaluation".
    """

    if mode not in {"training", "evaluation"}:
        raise ValueError(
            f"mode must be 'training' or 'evaluation', got {mode!r}"
        )

    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    if pred_steps < 1:
        raise ValueError("pred_steps must be at least 1")

    if stride < 1:
        raise ValueError("stride must be at least 1")

    data_feature = np.asarray(data_feature)
    data_target = np.asarray(data_target).reshape(-1, 1)

    if len(data_feature) != len(data_target):
        raise ValueError(
            "data_feature and data_target must have the same length"
        )

    feature_dim = data_feature.shape[1]

    if feature_dim + 1 != input_size:
        raise ValueError(
            f"Expected input_size={feature_dim + 1}, "
            f"but received input_size={input_size}"
        )

    number_of_samples = (
        len(data_feature) - window_size - pred_steps + 1
    )

    if number_of_samples <= 0:
        raise ValueError(
            "Not enough data to create sequences: "
            f"data length={len(data_feature)}, "
            f"window_size={window_size}, "
            f"pred_steps={pred_steps}"
        )

    X = []
    Y = []
    future_feature_sequences = []

    for start_idx in range(0, number_of_samples, stride):
        past_end_idx = start_idx + window_size
        future_end_idx = past_end_idx + pred_steps

        # Past external and auxiliary features
        past_features = data_feature[
            start_idx:past_end_idx
        ]

        # Past measured indoor PM2.5
        past_targets = data_target[
            start_idx:past_end_idx
        ]

        # Shape: [window_size, feature_dim + 1]
        past_window = np.concatenate(
            (past_features, past_targets),
            axis=1
        )

        # Shape: [pred_steps, 1]
        target_sequence = data_target[
            past_end_idx:future_end_idx
        ]

        # Shape: [pred_steps, feature_dim]
        future_features = data_feature[
            past_end_idx:future_end_idx
        ]

        X.append(past_window)
        Y.append(target_sequence)
        future_feature_sequences.append(future_features)

    X_3D = np.asarray(X, dtype=np.float32)
    Y_3D = np.asarray(Y, dtype=np.float32)
    future_features_3D = np.asarray(
        future_feature_sequences,
        dtype=np.float32
    )

    X_tensor = torch.from_numpy(X_3D)
    Y_tensor = torch.from_numpy(Y_3D)
    future_features_tensor = torch.from_numpy(
        future_features_3D
    )

    print("--- Created 3D Tensor ---")
    print(
        f"X: {X_tensor.shape} | "
        f"Y_seq: {Y_tensor.shape} | "
        f"Future_Feat: {future_features_tensor.shape} | "
        f"Stride: {stride}"
    )

    dataset = TensorDataset(
        X_tensor,
        Y_tensor,
        future_features_tensor
    )

    batch_size = batch_size_optimization(
        window_size=window_size,
        feature_dim=input_size
    )

    shuffle = mode == "training"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
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

    model.load_state_dict(checkpoint['trained_model_state_dict'])
    if regressor is not None and regressor_key in checkpoint:
        regressor.load_state_dict(checkpoint[regressor_key])

    print(f"--- Successfully loaded model from: {save_model_path} ---")

    model.to(device)
    if regressor is not None:
        regressor.to(device)

    return model, regressor
