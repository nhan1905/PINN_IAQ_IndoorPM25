import os
import torch
import numpy as np
import pandas as pd
from metrics import *
from mass_balance_equation import *


def _calculate_pm25_slopes(current_window, long_lag=5):
    """Return the latest and long-window indoor PM2.5 slopes."""
    if current_window.size(1) < 2:
        raise ValueError("At least two time steps are required for slope features")

    lag = min(long_lag, current_window.size(1))
    latest_pm25 = current_window[:, -1, -1]

    slope_1 = (
        latest_pm25 - current_window[:, -2, -1]
    ).unsqueeze(1)

    slope_5 = (
        (latest_pm25 - current_window[:, -lag, -1])
        / float(lag - 1)
    ).unsqueeze(1)

    return slope_1, slope_5

# ==========================================
# 2. EVALUATION CHO MB MODEL
# ==========================================
def evaluation_mb_physical_sensor(model, regressor, device, test_loader, result_file_name, save_path):
    print(f"--- Start Evaluating MB Physical Sensor on device: {device} ---")
    model.eval()
    regressor.eval()
    mb_equation = MassBalanceEquation(dt=1.0/60.0).to(device)
       
    all_y_true, all_y_pred = [], []
    all_aer, all_p, all_k, all_s = [], [], [], []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets_seq, future_feats) in enumerate(test_loader):         
            inputs, targets_seq, future_feats = inputs.to(device), targets_seq.to(device), future_feats.to(device)
            _, (hidden, _) = model(inputs) 
            latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            prev_c_indoor = inputs[:, -1, -1].unsqueeze(1)            
            c_outdoor = future_feats[:, 0, 0].unsqueeze(1)
            slope_1, slope_5 = _calculate_pm25_slopes(inputs)
            
            params_logit_seq = regressor(
                context=future_feats[:, 0, :],
                prev_c_indoor=prev_c_indoor,
                slope_1=slope_1,
                slope_5=slope_5,
                latent=latent
            )
                
            pred_raw, aer, p, k, s, _, _ = mb_equation(params_logit_seq, c_outdoor, prev_c_indoor)

            all_y_pred.append(pred_raw.cpu().numpy())
            all_y_true.append(targets_seq[:, 0, :].cpu().numpy())
            all_aer.append(aer.cpu().numpy())
            all_p.append(p.cpu().numpy())
            all_k.append(k.cpu().numpy())
            all_s.append(s.cpu().numpy())
                           
    y_pred = np.concatenate(all_y_pred).flatten()
    y_true = np.concatenate(all_y_true).flatten()
    aer_vals = np.concatenate(all_aer).flatten()
    p_vals = np.concatenate(all_p).flatten()
    k_vals = np.concatenate(all_k).flatten()
    s_vals = np.concatenate(all_s).flatten()
    
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Final Test R2 Score: {metrics['R2']:.4f}")

    results_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'AER': aer_vals, 'P': p_vals, 'K': k_vals, 'S': s_vals, **{k: [v]*len(y_true) for k, v in metrics.items()}})
    results_df.to_csv(os.path.join(save_path, result_file_name), index=False)
    return y_true, y_pred, aer_vals, p_vals, k_vals, s_vals

def evaluation_mb_virtual_sensor(model, regressor, device, test_loader, result_file_name, save_path):
    print(f"--- Start Evaluating MB Virtual Sensor (Multi-step) on device: {device} ---")
    model.eval()
    regressor.eval()
    mb_equation = MassBalanceEquation(dt=1.0/60.0).to(device)

    it = iter(test_loader)
    first_inputs, _, _ = next(it)
    current_window = first_inputs[0:1, :, :].to(device)

    all_y_true, all_features = [], []

    for inp, tar, feat in test_loader:
        all_features.append(feat.cpu().numpy()[:, 0, :])
        all_y_true.append(tar.cpu().numpy()[:, 0, :])
    
    full_features = torch.from_numpy(np.concatenate(all_features, axis=0)).to(device).float()
    full_ground_truth = np.concatenate(all_y_true, axis=0).flatten()

    all_y_pred, all_aer, all_p, all_k, all_s = [], [], [], [], []

    with torch.no_grad():
        for t in range(len(full_ground_truth)):
            _, (hidden, _) = model(current_window) 
            latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            prev_c_indoor = current_window[:, -1, -1].unsqueeze(1)
            c_outdoor = full_features[t:t+1, 0:1]
            slope_1, slope_5 = _calculate_pm25_slopes(current_window)
            params_logit_seq = regressor(
                context=full_features[t:t+1, :],
                prev_c_indoor=prev_c_indoor,
                slope_1=slope_1,
                slope_5=slope_5,
                latent=latent
            )
            
            pred_raw, aer, p, k, s, _, _ = mb_equation(params_logit_seq, c_outdoor, prev_c_indoor)

            all_y_pred.append(pred_raw.cpu().numpy())
            all_aer.append(aer.cpu().numpy()); all_p.append(p.cpu().numpy())
            all_k.append(k.cpu().numpy()); all_s.append(s.cpu().numpy())

            next_step_features = full_features[t, :].unsqueeze(0)
            new_row = torch.cat((next_step_features, pred_raw), dim=1).unsqueeze(1)
            current_window = torch.cat((current_window[:, 1:, :], new_row), dim=1)

    y_pred = np.concatenate(all_y_pred).flatten()
    y_true = full_ground_truth
    aer_vals, p_vals = np.concatenate(all_aer).flatten(), np.concatenate(all_p).flatten()
    k_vals, s_vals = np.concatenate(all_k).flatten(), np.concatenate(all_s).flatten()
    
    metrics = calculate_metrics(y_true, y_pred) 
    print(f"Virtual Sensor (MB) R2 Score: {metrics['R2']:.4f}")

    results_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'AER': aer_vals, 'P': p_vals, 'K': k_vals, 'S': s_vals, **{k: [v]*len(y_true) for k, v in metrics.items()}})
    results_df.to_csv(os.path.join(save_path, result_file_name), index=False)
    return y_true, y_pred, aer_vals, p_vals, k_vals, s_vals

def evaluation_mb_virtual_sensor_autoreset(
    model, regressor, device, test_loader,
    result_file_name, save_path, reset_period
):
    print(f"--- Evaluating MB Virtual Sensor with AUTO-RESET ({reset_period} steps) ---")

    model.eval()
    regressor.eval()
    mb_equation = MassBalanceEquation(dt=1.0/60.0).to(device)

    all_inputs = []
    all_future_features = []
    all_targets = []

    # test_loader phải được tạo với auto_reset=True
    # X:          [batch, window_size, 8]
    # Y_seq:      [batch, window_size, 1, 1]
    # FutureFeat: [batch, window_size, 7]
    for inputs, targets_seq, future_feats in test_loader:
        all_inputs.append(inputs.cpu().numpy())
        all_future_features.append(future_feats.cpu().numpy())
        all_targets.append(targets_seq.cpu().numpy())

    full_inputs = torch.from_numpy(
        np.concatenate(all_inputs, axis=0)
    ).to(device).float()

    full_future_features = torch.from_numpy(
        np.concatenate(all_future_features, axis=0)
    ).to(device).float()

    full_targets = np.concatenate(all_targets, axis=0)

    all_y_pred = []
    all_y_true = []
    all_aer = []
    all_p = []
    all_k = []
    all_s = []

    n_blocks = full_inputs.shape[0]

    with torch.no_grad():

        for b in range(n_blocks):

            # RESET: dùng actual window đầu mỗi block
            current_window = full_inputs[b:b+1, :, :]          # [1, 5, 8]
            block_features = full_future_features[b:b+1, :, :] # [1, 5, 7]
            block_targets = full_targets[b:b+1, :, :]       # [1, 5, 1]

            n_steps = block_features.shape[1]

            for t in range(n_steps):

                _, (hidden, _) = model(current_window)

                latent = torch.cat(
                    (hidden[-2, :, :], hidden[-1, :, :]),
                    dim=1
                )

                prev_c_indoor = current_window[:, -1, -1].unsqueeze(1)

                future_context = block_features[:, t, :]      # [1, 7]
                c_outdoor = block_features[:, t, 0:1]         # [1, 1]
                slope_1, slope_5 = _calculate_pm25_slopes(
                    current_window
                )

                params_t = regressor(
                    context=future_context,
                    prev_c_indoor=prev_c_indoor,
                    slope_1=slope_1,
                    slope_5=slope_5,
                    latent=latent
                )  # [1, 4]

                pred_raw, aer, p, k, s, _, _ = mb_equation(
                    params_t,
                    c_outdoor,
                    prev_c_indoor
                )

                all_y_pred.append(pred_raw.cpu().numpy())
                all_y_true.append(block_targets[:, t, :].reshape(1, 1))

                all_aer.append(aer.cpu().numpy())
                all_p.append(p.cpu().numpy())
                all_k.append(k.cpu().numpy())
                all_s.append(s.cpu().numpy())

                # update virtual window bằng predicted ID_PM25
                new_row = torch.cat(
                    (future_context, pred_raw),
                    dim=1
                ).unsqueeze(1)  # [1, 1, 8]

                current_window = torch.cat(
                    (current_window[:, 1:, :], new_row),
                    dim=1
                )

    y_pred = np.concatenate(all_y_pred).flatten()
    y_true = np.concatenate(all_y_true).flatten()

    aer_vals = np.concatenate(all_aer).flatten()
    p_vals = np.concatenate(all_p).flatten()
    k_vals = np.concatenate(all_k).flatten()
    s_vals = np.concatenate(all_s).flatten()

    metrics = calculate_metrics(y_true, y_pred)

    print(f"Virtual Sensor (MB) R2 Score with RESET: {metrics['R2']:.4f}")

    results_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'AER': aer_vals,
        'P': p_vals,
        'K': k_vals,
        'S': s_vals,
        **{k: [v] * len(y_true) for k, v in metrics.items()}
    })

    results_df.to_csv(
        os.path.join(save_path, result_file_name),
        index=False
    )

    return y_true, y_pred, aer_vals, p_vals, k_vals, s_vals
