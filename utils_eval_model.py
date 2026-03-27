# 1. Import libraries\
import os
import torch
import numpy as np
import pandas as pd
from metrics import calculate_astm_metrics

# Evaluate the normal model
def evaluation_pure_physical_sensor(model, regressor, device, test_loader, scaler_indoor, result_file_name, save_path):

    print(f"--- Start Evaluating Pure Physical Sensor Model (w/o Mass Balance Informed) on device: {device} ---")

    model.eval()
    regressor.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            
            # Forward through Encoder + Regressor
            _, (hidden, _) = model.encoder(inputs) 
            latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            predictions = regressor(latent)
            
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate all batches
    y_pred_scaled = np.concatenate(all_preds)
    y_pred = scaler_indoor.inverse_transform(y_pred_scaled)

    y_true = np.concatenate(all_targets)
    y_true = scaler_indoor.inverse_transform(y_true)

    #  Calculate R2 on the original unit
    metrics = calculate_astm_metrics(y_true, y_pred)
    print(f"Final Test R2 Score: {metrics['R2']:.4f}")

    #  Save the evaluation results to a CSV file
    results_df = pd.DataFrame({
                                'y_true': y_true.flatten(),
                                'y_pred': y_pred.flatten(),
                                'MSE': metrics['MSE'],
                                'RMSE': metrics['RMSE'],
                                'MAE': metrics['MAE'],
                                'MAPE': metrics['MAPE'],
                                'R': metrics['R'],
                                'R2': metrics['R2'],
                                'b': metrics['b'],
                                'a': metrics['a'],
                                'NMSE': metrics['NMSE'],
                                'FB': metrics['FB'],
                                'FS': metrics['FS']
                                })
    
    output_path = os.path.join(save_path, result_file_name)
    results_df.to_csv(output_path, index=False)
    print(f"--- Successfully saved evaluation results to: {output_path} ---")

    return y_true, y_pred

# Evaluate the normal model with Autoregressive Forecasting
def evaluation_pure_virtual_sensor(model, regressor, device, test_loader, scaler_indoor, result_file_name, save_path):
    
    print(f"--- Start Evaluating Pure Virtual Sensor Model (w/o Mass Balance Informed) on device: {device} ---")
   
    model.eval()
    regressor.eval()

    all_y_true_scaled = []
    all_features = []
    
    with torch.no_grad():
        for inp, tar in test_loader:
            all_y_true_scaled.append(tar.numpy())
            all_features.append(inp.numpy())
            
    y_true_scaled = np.concatenate(all_y_true_scaled, axis=0)
    features_full = np.concatenate(all_features, axis=0) 

    init_features = torch.from_numpy(features_full[0:1, :, :-1]).to(device).float()

    init_indoor = torch.from_numpy(y_true_scaled[0:5]).reshape(1, 5, 1).to(device).float()

    current_window = torch.cat((init_features, init_indoor), dim=2)
    
    virtual_preds = []

    with torch.no_grad():
        for i in range(len(y_true_scaled)):
            _, (hidden, _) = model.encoder(current_window) 
            latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            pred_raw = regressor(latent)
                      
            virtual_preds.append(pred_raw.cpu().numpy())

            if i < len(y_true_scaled) - 1:
                next_features = torch.from_numpy(features_full[i+1, -1:, :-1]).to(device).float()
                
                pred_rescaled = _scaler_transform(scaler_indoor, pred_raw.cpu().numpy().reshape(-1, 1))
                pred_rescaled = torch.as_tensor(pred_rescaled, device=device, dtype=torch.float32)
                
                next_step_data = torch.cat((next_features, pred_rescaled), dim=1)
                
                current_window = torch.cat((current_window[:, 1:, :], next_step_data.unsqueeze(0)), dim=1)

    y_pred = np.concatenate(virtual_preds).flatten()
    y_true = _scaler_inverse_transform(scaler_indoor, y_true_scaled.reshape(-1, 1)).flatten()
     
    # 5. Calculate the ASTM D 5157 metrics
    metrics = calculate_astm_metrics(y_true, y_pred)
    print(f"Virtual Sensor (PURE) R2 Score: {metrics['R2']:.4f}")

    results_df = pd.DataFrame({
                                'y_true': y_true,
                                'y_pred': y_pred,
                                'MSE': metrics['MSE'],
                                'RMSE': metrics['RMSE'],
                                'MAE': metrics['MAE'],
                                'MAPE': metrics['MAPE'],
                                'R': metrics['R'],
                                'R2': metrics['R2'],
                                'b': metrics['b'],
                                'a': metrics['a'],
                                'NMSE': metrics['NMSE'],
                                'FB': metrics['FB'],
                                'FS': metrics['FS']
                                })
    
    output_path = os.path.join(save_path, result_file_name)
    results_df.to_csv(output_path, index=False)
    print(f"--- Successfully saved evaluation results to: {output_path} ---")

    return y_true, y_pred

# 4. Evaluate the pure model with Autoregressive Forecasting with Periodic Reset
def evaluation_pure_virtual_sensor_autoreset(model, regressor, device, test_loader, scaler_indoor, result_file_name, save_path, reset_period=5):
    
    print(f"--- Start Evaluating Virtual Sensor (PURE Model) with Periodic Reset (Time Steps) on device: {device} ---")
    
    model.eval()
    regressor.eval()

    all_y_true_scaled = []
    all_features = []
    with torch.no_grad():
        for inp, tar in test_loader:
            all_y_true_scaled.append(tar.numpy())
            all_features.append(inp.numpy())
            
    y_true_scaled = np.concatenate(all_y_true_scaled, axis=0)
    features_full = np.concatenate(all_features, axis=0)
    window_size = features_full.shape[1] 

    virtual_preds = []

    init_features = torch.from_numpy(features_full[0:1, :, :-1]).to(device).float()
    end_idx = min(window_size, len(y_true_scaled))
    actual_indoor_seq = y_true_scaled[0:end_idx]
    if len(actual_indoor_seq) < window_size:
        padding = np.repeat(actual_indoor_seq[-1:], window_size - len(actual_indoor_seq), axis=0)
        actual_indoor_seq = np.concatenate([actual_indoor_seq, padding], axis=0)
    init_indoor = torch.from_numpy(actual_indoor_seq).reshape(1, window_size, 1).to(device).float()
    current_window = torch.cat((init_features, init_indoor), dim=2)

    with torch.no_grad():
        for i in range(len(y_true_scaled)):
            
            if i % reset_period == 0:
                init_features = torch.from_numpy(features_full[i:i+1, :, :-1]).to(device).float()
                
                end_idx = min(i + window_size, len(y_true_scaled))
                actual_indoor_seq = y_true_scaled[i:end_idx]
                
                if len(actual_indoor_seq) < window_size:
                    padding = np.repeat(actual_indoor_seq[-1:], window_size - len(actual_indoor_seq), axis=0)
                    actual_indoor_seq = np.concatenate([actual_indoor_seq, padding], axis=0)
                
                init_indoor = torch.from_numpy(actual_indoor_seq).reshape(1, window_size, 1).to(device).float()
                
                current_window = torch.cat((init_features, init_indoor), dim=2)

            _, (hidden, _) = model.encoder(current_window) 
            latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            pred_raw = regressor(latent)
                     
            virtual_preds.append(pred_raw.cpu().numpy())

            if i < len(y_true_scaled) - 1:
                
                if (i + 1) % reset_period == 0:
                    continue
                
                next_features = torch.from_numpy(features_full[i+1, -1:, :-1]).to(device).float()
                
                pred_rescaled = _scaler_transform(scaler_indoor, pred_raw.cpu().numpy().reshape(-1, 1))
                pred_rescaled = torch.as_tensor(pred_rescaled, device=device, dtype=torch.float32)
                
                next_step_data = torch.cat((next_features, pred_rescaled), dim=1)
                
                current_window = torch.cat((current_window[:, 1:, :], next_step_data.unsqueeze(0)), dim=1)

    y_pred = np.concatenate(virtual_preds).flatten()
    y_true = _scaler_inverse_transform(scaler_indoor, y_true_scaled.reshape(-1, 1)).flatten()
    
    metrics = calculate_astm_metrics(y_true, y_pred)
    print(f"Virtual Sensor (PURE) R2 Score with Reset: {metrics['R2']:.4f}")

    results_df = pd.DataFrame({
                                'y_true': y_true,
                                'y_pred': y_pred,
                                'MSE': metrics['MSE'],
                                'RMSE': metrics['RMSE'],
                                'MAE': metrics['MAE'],
                                'MAPE': metrics['MAPE'],
                                'R': metrics['R'],
                                'R2': metrics['R2'],
                                'b': metrics['b'],
                                'a': metrics['a'],
                                'NMSE': metrics['NMSE'],
                                'FB': metrics['FB'],
                                'FS': metrics['FS']
    })
    output_path = os.path.join(save_path, result_file_name)
    results_df.to_csv(output_path, index=False)
    print(f"--- Successfully saved evaluation results to: {output_path} ---")

    return y_true, y_pred

# C. FUNCTION DEFINES FOR MASS BALANCE INFORMED MODELS
def _scaler_transform(scaler, X):
    X = np.asarray(X)
    if getattr(scaler, 'feature_names_in_', None) is not None:
        X = pd.DataFrame(X, columns=scaler.feature_names_in_)
    return scaler.transform(X)

def _scaler_inverse_transform(scaler, X):
    X = np.asarray(X)
    if getattr(scaler, 'feature_names_in_', None) is not None:
        X = pd.DataFrame(X, columns=scaler.feature_names_in_)
    return scaler.inverse_transform(X)

# Define the Mass Balance Equation
def MassBalanceEquation(params_logit, scaler_outdoor, scaler_indoor, avg_outdoor_scaled, prev_indoor_scaled, dt=1.0):

    # 0.1 Decode 4 physical parameters (Physical Constraints)
    # AER (Air Exchange Rate): > 0. Usually from 0.1 to 5.0 (times/hour)
    AER_t = 5.0 * torch.sigmoid(params_logit[:, 0].unsqueeze(1))
    
    # P (Penetration Efficiency): 0 <= P <= 1
    P_t = torch.sigmoid(params_logit[:, 1].unsqueeze(1))
    
    # K (Deposition Rate): > 0. Usually from 0.1 to 1.0 (1/hour)
    K_t = 2.0 * torch.sigmoid(params_logit[:, 2].unsqueeze(1))
    
    # S (Source Generation Rate): >= 0 (µg/m3/h)
    S_t = 50.0 * torch.sigmoid(params_logit[:, 3].unsqueeze(1))

    # Un-scale Input Data to the original unit
    C_in_prev = _scaler_inverse_transform(scaler_indoor, prev_indoor_scaled.cpu().detach().numpy())
    C_out = _scaler_inverse_transform(scaler_outdoor, avg_outdoor_scaled.cpu().detach().numpy())
    
    # Convert back to torch tensors on the same device
    C_in_prev = torch.tensor(C_in_prev, dtype=torch.float32, device=prev_indoor_scaled.device)
    C_out = torch.tensor(C_out, dtype=torch.float32, device=avg_outdoor_scaled.device)

    # Standard Mass Balance Formula
    # C_pred = [C_in_prev + dt * (AER * P * C_out + S)] / [1 + dt * (AER + K)]
    
    numerator = C_in_prev + dt * (AER_t * P_t * C_out + S_t)
    denominator = 1.0 + dt * (AER_t + K_t)
    
    pred_raw = numerator / denominator
    
    return pred_raw, AER_t, P_t, K_t, S_t

# Evaluate the mass balance informed model
def evaluation_mb_physical_sensor(model, regressor, device, test_loader, scaler_outdoor, scaler_indoor, result_file_name, save_path):
    
    print(f"--- Start Evaluating Mass Balance Physical Sensor Model (w/ Mass Balance Informed) on device: {device} ---")
    
    model.eval()
    regressor.eval()
       
    all_preds = []
    all_targets = []
    all_aer, all_p, all_k, all_s = [], [], [], []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            
            _, (hidden, _) = model.encoder(inputs) 
            latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            params_logit = regressor(latent)
            
            avg_outdoor_scaled = torch.mean(inputs[:, :, 0], dim=1).unsqueeze(1)
            prev_indoor_scaled = inputs[:, -2, -1].unsqueeze(1)
            
            pred_raw, aer, p, k, s = MassBalanceEquation(params_logit, scaler_outdoor, scaler_indoor, avg_outdoor_scaled, prev_indoor_scaled)
            target_raw = _scaler_inverse_transform(scaler_indoor, targets.cpu().detach().numpy())
            
            all_preds.append(pred_raw.cpu().numpy())
            all_targets.append(target_raw)
            all_aer.append(aer.cpu().numpy())
            all_p.append(p.cpu().numpy())
            all_k.append(k.cpu().numpy())
            all_s.append(s.cpu().numpy())

    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_targets).flatten()
    aer_vals = np.concatenate(all_aer).flatten()
    p_vals = np.concatenate(all_p).flatten()
    k_vals = np.concatenate(all_k).flatten()
    s_vals = np.concatenate(all_s).flatten()
    
    metrics = calculate_astm_metrics(y_true, y_pred)
    print(f"Final Test R2 Score: {metrics['R2']:.4f}")

    # Save the evaluation results to a CSV file
    results_df = pd.DataFrame({
                                'y_true': y_true.flatten(),
                                'y_pred': y_pred.flatten(),
                                'AER': aer_vals.flatten(),
                                'P': p_vals.flatten(),
                                'K': k_vals.flatten(),
                                'S': s_vals.flatten(),
                                'MSE': metrics['MSE'],
                                'RMSE': metrics['RMSE'],
                                'MAE': metrics['MAE'],
                                'MAPE': metrics['MAPE'],
                                'R': metrics['R'],
                                'R2': metrics['R2'],
                                'b': metrics['b'],
                                'a': metrics['a'],
                                'NMSE': metrics['NMSE'],
                                'FB': metrics['FB'],
                                'FS': metrics['FS']
                                })
    
    output_path = os.path.join(save_path, result_file_name)
    results_df.to_csv(output_path, index=False)
    print(f"--- Successfully saved evaluation results to: {output_path} ---")

    return y_true, y_pred, aer_vals, p_vals, k_vals, s_vals

# Evaluate the mass balance informed model with Autoregressive Forecasting
def evaluation_mb_virtual_sensor(model, regressor, device, test_loader, scaler_outdoor, scaler_indoor, result_file_name, save_path):
    print(f"--- Start Evaluating Mass Balance Virtual Sensor Model (w/ Mass Balance Informed) on device: {device} ---")
    model.eval()
    regressor.eval()

    all_y_true_scaled = []
    all_features = []
    
    with torch.no_grad():
        for inp, tar in test_loader:
            all_y_true_scaled.append(tar.numpy())
            all_features.append(inp.numpy())
            
    y_true_scaled = np.concatenate(all_y_true_scaled, axis=0)
    features_full = np.concatenate(all_features, axis=0) 

    # [1, 5, num_features]
    init_features = torch.from_numpy(features_full[0:1, :, :-1]).to(device).float()

    init_indoor = torch.from_numpy(y_true_scaled[0:5]).reshape(1, 5, 1).to(device).float()

    current_window = torch.cat((init_features, init_indoor), dim=2)
    
    virtual_preds = []
    all_aer, all_p, all_k, all_s = [], [], [], []

    with torch.no_grad():
        for i in range(len(y_true_scaled)):
            _, (hidden, _) = model.encoder(current_window) 
            latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            params_logit = regressor(latent)
            
            avg_outdoor_scaled = torch.mean(current_window[:, :, 0], dim=1).unsqueeze(1)
            
            prev_indoor_scaled = current_window[:, -1, -1].unsqueeze(1)
            
            pred_raw, aer, p, k, s = MassBalanceEquation(
                                                            params_logit, 
                                                            scaler_outdoor, 
                                                            scaler_indoor, 
                                                            avg_outdoor_scaled, 
                                                            prev_indoor_scaled
                                                            )
            
            virtual_preds.append(pred_raw.cpu().numpy())
            all_aer.append(aer.cpu().numpy())
            all_p.append(p.cpu().numpy())
            all_k.append(k.cpu().numpy())
            all_s.append(s.cpu().numpy())

            if i < len(y_true_scaled) - 1:
                next_features = torch.from_numpy(features_full[i+1, -1:, :-1]).to(device).float()
                
                pred_rescaled = _scaler_transform(scaler_indoor, pred_raw.cpu().numpy().reshape(-1, 1))
                pred_rescaled = torch.as_tensor(pred_rescaled, device=device, dtype=torch.float32)
                
                next_step_data = torch.cat((next_features, pred_rescaled), dim=1)
                
                
                current_window = torch.cat((current_window[:, 1:, :], next_step_data.unsqueeze(0)), dim=1)

    y_pred = np.concatenate(virtual_preds).flatten()
    y_true = _scaler_inverse_transform(scaler_indoor, y_true_scaled.reshape(-1, 1)).flatten()
    
    aer_vals = np.concatenate(all_aer).flatten()
    p_vals = np.concatenate(all_p).flatten()
    k_vals = np.concatenate(all_k).flatten()
    s_vals = np.concatenate(all_s).flatten()
    
    metrics = calculate_astm_metrics(y_true, y_pred) 
    print(f"Virtual Sensor (MB) R2 Score: {metrics['R2']:.4f}")

    results_df = pd.DataFrame({
                                'y_true': y_true.flatten(),
                                'y_pred': y_pred.flatten(),
                                'AER': aer_vals.flatten(),
                                'P': p_vals.flatten(),
                                'K': k_vals.flatten(),
                                'S': s_vals.flatten(),
                                'MSE': metrics['MSE'],
                                'RMSE': metrics['RMSE'],
                                'MAE': metrics['MAE'],
                                'MAPE': metrics['MAPE'],
                                'R': metrics['R'],
                                'R2': metrics['R2'],
                                'b': metrics['b'],
                                'a': metrics['a'],
                                'NMSE': metrics['NMSE'],
                                'FB': metrics['FB'],
                                'FS': metrics['FS']
                                })
    
    output_path = os.path.join(save_path, result_file_name)
    results_df.to_csv(output_path, index=False)
    print(f"--- Successfully saved evaluation results to: {output_path} ---")

    return y_true, y_pred, aer_vals, p_vals, k_vals, s_vals

# Evaluate the mass balance informed model with Autoregressive Forecasting with Periodic Reset
def evaluation_mb_virtual_sensor_autoreset(model, regressor, device, test_loader, scaler_outdoor, scaler_indoor, result_file_name, save_path, reset_period=5):
    print(f"--- Start Virtual Sensor (MB Model) with Periodic Reset (Time Steps) ---")
    model.eval()
    regressor.eval()

    all_y_true_scaled = []
    all_features = []
    with torch.no_grad():
        for inp, tar in test_loader:
            all_y_true_scaled.append(tar.numpy())
            all_features.append(inp.numpy())
            
    y_true_scaled = np.concatenate(all_y_true_scaled, axis=0)
    features_full = np.concatenate(all_features, axis=0)
    window_size = features_full.shape[1] 

    virtual_preds = []
    all_aer, all_p, all_k, all_s = [], [], [], []

    init_features = torch.from_numpy(features_full[0:1, :, :-1]).to(device).float()
    end_idx = min(window_size, len(y_true_scaled))
    actual_indoor_seq = y_true_scaled[0:end_idx]
    if len(actual_indoor_seq) < window_size:
        padding = np.repeat(actual_indoor_seq[-1:], window_size - len(actual_indoor_seq), axis=0)
        actual_indoor_seq = np.concatenate([actual_indoor_seq, padding], axis=0)
    init_indoor = torch.from_numpy(actual_indoor_seq).reshape(1, window_size, 1).to(device).float()
    current_window = torch.cat((init_features, init_indoor), dim=2)

    with torch.no_grad():
        for i in range(len(y_true_scaled)):
            
            # i % reset_period == 0
            if i % reset_period == 0:
                # (Outdoor, Temp, RH, CO2) — shape [1, window_size, num_features-1]
                init_features = torch.from_numpy(features_full[i:i+1, :, :-1]).to(device).float()
                
                
                end_idx = min(i + window_size, len(y_true_scaled))
                actual_indoor_seq = y_true_scaled[i:end_idx]
                
                if len(actual_indoor_seq) < window_size:
                    padding = np.repeat(actual_indoor_seq[-1:], window_size - len(actual_indoor_seq), axis=0)
                    actual_indoor_seq = np.concatenate([actual_indoor_seq, padding], axis=0)
                
                init_indoor = torch.from_numpy(actual_indoor_seq).reshape(1, window_size, 1).to(device).float()
                
                current_window = torch.cat((init_features, init_indoor), dim=2)

            _, (hidden, _) = model.encoder(current_window) 
            latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            params_logit = regressor(latent)
            
            avg_outdoor_scaled = torch.mean(current_window[:, :, 0], dim=1).unsqueeze(1)
            prev_indoor_scaled = current_window[:, -1, -1].unsqueeze(1)
            
            pred_raw, aer, p, k, s = MassBalanceEquation(
                params_logit, scaler_outdoor, scaler_indoor, avg_outdoor_scaled, prev_indoor_scaled
            )
            
            virtual_preds.append(pred_raw.cpu().numpy())
            all_aer.append(aer.cpu().numpy())
            all_p.append(p.cpu().numpy())
            all_k.append(k.cpu().numpy())
            all_s.append(s.cpu().numpy())

            if i < len(y_true_scaled) - 1:
                if (i + 1) % reset_period == 0:
                    continue
                
                next_features = torch.from_numpy(features_full[i+1, -1:, :-1]).to(device).float()
                
                pred_rescaled = _scaler_transform(scaler_indoor, pred_raw.cpu().numpy().reshape(-1, 1))
                pred_rescaled = torch.as_tensor(pred_rescaled, device=device, dtype=torch.float32)
                
                next_step_data = torch.cat((next_features, pred_rescaled), dim=1)
                
                current_window = torch.cat((current_window[:, 1:, :], next_step_data.unsqueeze(0)), dim=1)

    y_pred = np.concatenate(virtual_preds).flatten()
    y_true = _scaler_inverse_transform(scaler_indoor, y_true_scaled.reshape(-1, 1)).flatten()
    
    aer_vals = np.concatenate(all_aer).flatten()
    p_vals = np.concatenate(all_p).flatten()
    k_vals = np.concatenate(all_k).flatten()
    s_vals = np.concatenate(all_s).flatten()
    
    metrics = calculate_astm_metrics(y_true, y_pred) 
    print(f"Virtual Sensor (MB) R2 Score with Reset: {metrics['R2']:.4f}")

    results_df = pd.DataFrame({
                                'y_true': y_true.flatten(),
                                'y_pred': y_pred.flatten(),
                                'AER': aer_vals.flatten(),
                                'P': p_vals.flatten(),
                                'K': k_vals.flatten(),
                                'S': s_vals.flatten(),
                                'MSE': metrics['MSE'],
                                'RMSE': metrics['RMSE'],
                                'MAE': metrics['MAE'],
                                'MAPE': metrics['MAPE'],
                                'R': metrics['R'],
                                'R2': metrics['R2'],
                                'b': metrics['b'],
                                'a': metrics['a'],
                                'NMSE': metrics['NMSE'],
                                'FB': metrics['FB'],
                                'FS': metrics['FS']
                                })
    output_path = os.path.join(save_path, result_file_name)
    results_df.to_csv(output_path, index=False)
    print(f"--- Successfully saved evaluation results to: {output_path} ---")
    
    return y_true, y_pred, aer_vals, p_vals, k_vals, s_vals
