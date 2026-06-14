# Author: Nguyen Tran Nhat Minh (Revised for Autoregressive Evaluation Timeline)
import os
import torch
import numpy as np
import pandas as pd
from metrics import *
from mass_balance_equation import *

# ==========================================
# 1. EVALUATION OF PURE MODEL             
# ==========================================
def evaluation_pure_physical_sensor(model, regressor, device, test_loader, steps_ahead, result_file_name, save_path):
    print(f"--- Start Evaluating PURE BiLSTM (Timeline: {steps_ahead} steps) ---")
    model.eval()
    regressor.eval()
    
    all_y_true_full, all_y_pred_full = [], [] 
    y_true_for_metrics, y_pred_for_metrics = [], [] 
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, future_feats) in enumerate(test_loader):
            inputs, targets, future_feats = inputs.to(device), targets.to(device), future_feats.to(device)

            _, (hidden, _) = model.encoder(inputs)
            latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

            batch_y_pred, batch_y_true = [], []
            for t in range(steps_ahead):
                pred_t = regressor(latent, future_feats[:, t, :])
                batch_y_pred.append(pred_t.cpu().numpy())
                batch_y_true.append(targets[:, t, :].cpu().numpy())

            y_pred_seq_np = np.concatenate(batch_y_pred, axis=1)
            y_true_seq_np = np.concatenate(batch_y_true, axis=1)

            window_true = inputs[:, :, -1].cpu().numpy()
            nan_window = np.full_like(window_true, np.nan)

            full_y_true = np.concatenate([window_true, y_true_seq_np], axis=1)
            full_y_pred = np.concatenate([nan_window, y_pred_seq_np], axis=1)

            all_y_true_full.append(full_y_true)
            all_y_pred_full.append(full_y_pred)

            y_true_for_metrics.append(y_true_seq_np)
            y_pred_for_metrics.append(y_pred_seq_np)

    y_true_final = np.concatenate(all_y_true_full).flatten()
    y_pred_final = np.concatenate(all_y_pred_full).flatten()
    
    y_true_m = np.concatenate(y_true_for_metrics).flatten()
    y_pred_m = np.concatenate(y_pred_for_metrics).flatten()
    metrics = calculate_metrics(y_true_m, y_pred_m)
    print(f"Final Pure Multi-step R2 Score: {metrics['R2']:.4f}")

    results_df = pd.DataFrame({
        'y_true': y_true_final, 'y_pred': y_pred_final,
        'AER': [np.nan]*len(y_true_final), 'P': [np.nan]*len(y_true_final),
        'K':   [np.nan]*len(y_true_final), 'S': [np.nan]*len(y_true_final),
        **{k: [v]*len(y_true_final) for k, v in metrics.items()}
    })
    
    if not os.path.exists(save_path): os.makedirs(save_path)
    results_df.to_csv(os.path.join(save_path, result_file_name), index=False)
    
    return y_true_final, y_pred_final, None, None, None, None


# ==========================================
# 2. EVALUATION OF MB MODEL
# ==========================================
def evaluation_mb_physical_sensor(model, regressor, device, test_loader, steps_ahead, result_file_name, save_path):
    print(f"--- Start Evaluating MB Physical Sensor (Multi-step Timeline: {steps_ahead} steps) ---")
    model.eval()
    regressor.eval()
    mb_equation = MassBalanceEquation(dt=5.0/60.0).to(device)
       
    all_y_true_full, all_y_pred_full = [], []
    all_aer, all_p, all_k, all_s = [], [], [], []
    y_true_for_metrics, y_pred_for_metrics = [], []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, future_feats) in enumerate(test_loader):
            inputs, targets, future_feats = inputs.to(device), targets.to(device), future_feats.to(device)

            _, (hidden, _) = model(inputs)
            latent = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

            prev_c_indoor = inputs[:, -1, -1].unsqueeze(1)

            batch_y_pred, batch_y_true = [], []
            batch_aer, batch_p, batch_k, batch_s = [], [], [], []

            for t in range(targets.size(1)):
                current_context_t = future_feats[:, t, :]
                params_logit_t = regressor(current_context_t, prev_c_indoor, latent)
                
                if len(params_logit_t.shape) == 3:
                    params_logit_t = params_logit_t[:, 0, :]
                
                c_outdoor_t = future_feats[:, t, 0].unsqueeze(1)
                
                pred_raw, aer, p, k, s, _, _ = mb_equation(params_logit_t, c_outdoor_t, prev_c_indoor)

                batch_y_pred.append(pred_raw.cpu().numpy())
                batch_y_true.append(targets[:, t, :].cpu().numpy())
                batch_aer.append(aer.cpu().numpy())
                batch_p.append(p.cpu().numpy())
                batch_k.append(k.cpu().numpy())
                batch_s.append(s.cpu().numpy())
                
                prev_c_indoor = pred_raw 

            window_true = inputs[:, :, -1].cpu().numpy() 
            nan_window = np.full_like(window_true, np.nan) 

            y_pred_seq = np.stack(batch_y_pred, axis=1).reshape(inputs.shape[0], -1)
            y_true_seq = np.stack(batch_y_true, axis=1).reshape(inputs.shape[0], -1)
            
            full_y_true = np.concatenate([window_true, y_true_seq], axis=1)
            full_y_pred = np.concatenate([nan_window, y_pred_seq], axis=1)
            
            all_y_true_full.append(full_y_true)
            all_y_pred_full.append(full_y_pred)
            
            y_true_for_metrics.append(y_true_seq)
            y_pred_for_metrics.append(y_pred_seq)

            aer_seq = np.stack(batch_aer, axis=1).reshape(inputs.shape[0], -1)
            all_aer.append(np.concatenate([nan_window, aer_seq], axis=1))
            
            p_seq = np.stack(batch_p, axis=1).reshape(inputs.shape[0], -1)
            all_p.append(np.concatenate([nan_window, p_seq], axis=1))
            
            k_seq = np.stack(batch_k, axis=1).reshape(inputs.shape[0], -1)
            all_k.append(np.concatenate([nan_window, k_seq], axis=1))
            
            s_seq = np.stack(batch_s, axis=1).reshape(inputs.shape[0], -1)
            all_s.append(np.concatenate([nan_window, s_seq], axis=1))

    y_true_final = np.concatenate(all_y_true_full).flatten()
    y_pred_final = np.concatenate(all_y_pred_full).flatten()
    aer_final = np.concatenate(all_aer).flatten()
    p_final = np.concatenate(all_p).flatten()
    k_final = np.concatenate(all_k).flatten()
    s_final = np.concatenate(all_s).flatten()
    
    y_true_m = np.concatenate(y_true_for_metrics).flatten()
    y_pred_m = np.concatenate(y_pred_for_metrics).flatten()
    metrics = calculate_metrics(y_true_m, y_pred_m)
    print(f"Final Test R2 Score ({steps_ahead}-step): {metrics['R2']:.4f}")

    results_df = pd.DataFrame({
        'y_true': y_true_final, 'y_pred': y_pred_final, 
        'AER': aer_final, 'P': p_final, 'K': k_final, 'S': s_final, 
        **{k: [v]*len(y_true_final) for k, v in metrics.items()}
    })
    
    if not os.path.exists(save_path): os.makedirs(save_path)
    results_df.to_csv(os.path.join(save_path, result_file_name), index=False)
    
    return y_true_final, y_pred_final, aer_final, p_final, k_final, s_final