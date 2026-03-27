# 1. Import libraries
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from loss_function import MassBalanceLoss

# B. FUNCTION DEFINE FOR NORMAL MODELS (without Mass Balance Informed)
# 1. Train the model
def training_pure_model(
                    model,
                    regressor,
                    device,
                    scaler_indoor,
                    train_loader,   # (batch_size, input_size, window_size)
                    val_loader, 
                    learning_rate,
                    epochs,
                    model_file_name,
                    save_path
                    ):

    print(f"--- Start Training on device: {device} ---")
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(regressor.parameters()), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        # Training phase
        model.train()
        regressor.train()
        total_train_loss = 0

        all_train_preds = []
        all_train_targets = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward through BiLSTM
            _, (hidden, _) = model.encoder(inputs) 
            latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            
            # Pass through the prediction layer
            predictions = regressor(latent)
            
            # Calculate loss (compared to actual ID_PM25)
            train_loss = criterion(predictions, targets)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            total_train_loss += train_loss.item()

            predictions_raw = scaler_indoor.inverse_transform(predictions.cpu().detach().numpy())
            targets_raw = scaler_indoor.inverse_transform(targets.cpu().detach().numpy())

            all_train_preds.append(predictions_raw)
            all_train_targets.append(targets_raw)

        avg_train_loss = total_train_loss / len(train_loader)

        train_r2 = r2_score(np.concatenate(all_train_targets), np.concatenate(all_train_preds))

        # Validation phase
        model.eval()
        regressor.eval()
        total_val_loss = 0.0

        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Get features from pre-trained encoder
                # Forward through encoder
                _, (hidden, _) = model.encoder(inputs) 
                latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
                
                # Pass through the prediction layer
                predictions = regressor(latent)

                predictions_raw = scaler_indoor.inverse_transform(predictions.cpu().detach().numpy())
                targets_raw = scaler_indoor.inverse_transform(targets.cpu().detach().numpy())
                
                # Calculate loss (compared to actual ID_PM25)
                val_loss = criterion(predictions, targets)

                total_val_loss += val_loss.item()

                all_val_preds.append(predictions_raw)
                all_val_targets.append(targets_raw)

        avg_val_loss = total_val_loss / len(val_loader)

        val_r2 = r2_score(np.concatenate(all_val_targets), np.concatenate(all_val_preds))

        print(f"Training Epoch [{epoch+1}/{epochs}] - Training MSE: {avg_train_loss:.6f} - Training R2: {train_r2:.6f} - Validation MSE: {avg_val_loss:.6f} - Validation R2: {val_r2:.6f}")

    # Save the trained model
    torch.save({
                'trained_encoder_state_dict': model.encoder.state_dict(),
                'trained_regressor_state_dict': regressor.state_dict()}, 
                os.path.join(save_path, model_file_name)
                )
    
    print(f"\n--- Successfully saved trained model to: {os.path.join(save_path, model_file_name)} ---")

    return model, regressor

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

# 0. Define the Mass Balance Equation
def MassBalanceEquation(params_logit, scaler_outdoor, scaler_indoor, avg_outdoor_scaled, prev_indoor_scaled, dt=1.0):
    """
    Input: params_logit (Batch, 4) -> [AER, P, K, S]
    """
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
    
    # ADDED C_in_prev and C_out to the return statement
    return pred_raw, AER_t, P_t, K_t, S_t, C_in_prev, C_out

# Define Training Function for Mass Balance Informed Model
def training_mb_model(  model, 
                        regressor,
                        device, 
                        scaler_outdoor,
                        scaler_indoor,
                        train_loader,   # (batch_size, input_size, window_size)
                        val_loader, 
                        learning_rate, 
                        epochs,
                        model_file_name,
                        save_path
                        ):
    print(f"--- Start Training Mass Balance Informed Model on device: {device} ---")
    
    optimizer = torch.optim.Adam([
                                    {'params': model.parameters(), 'lr': learning_rate},     
                                    {'params': regressor.parameters(), 'lr': learning_rate}
                                    ], 
                                    weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    criterion = MassBalanceLoss(dt=1.0, epsilon=0.001, lambda_mb=1.0, lambda_stab=0.5)

    print(f"--- Config: Epochs={epochs} | STRATEGY: PHYSICS MASS BALANCE (AER, P, K, S) ---")

    for epoch in range(epochs):
        model.train()
        regressor.train()
        total_loss = 0
        total_L_data = 0
        total_L_mb = 0
        total_L_stab = 0

        # Tracking variables
        avg_aer, avg_p, avg_k, avg_s = 0, 0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            _, (hidden, _) = model.encoder(inputs) 
            latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            params_logit = regressor(latent)
            
            avg_outdoor_scaled = torch.mean(inputs[:, :, 0], dim=1).unsqueeze(1)
            prev_indoor_scaled = inputs[:, -2, -1].unsqueeze(1)
            
            # Forward pass through Mass Balance Equation (UPDATED to unpack c_prev and c_out)
            pred_raw, aer, p, k, s, c_prev, c_out = MassBalanceEquation(params_logit, scaler_outdoor, scaler_indoor, avg_outdoor_scaled, prev_indoor_scaled)
            targets_raw = _scaler_inverse_transform(scaler_indoor, targets.cpu().detach().numpy())
            targets_raw = torch.tensor(targets_raw, dtype=torch.float32, device=targets.device)
            
            # Mass Balance Loss (UPDATED to pass c_prev and c_out to the criterion)
            loss, L_data, L_mb, L_stab = criterion(pred_raw, targets_raw, aer, p, k, s, c_prev, c_out)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_L_data += L_data.item()
            total_L_mb += L_mb.item()
            total_L_stab += L_stab.item()
            avg_aer += aer.mean().item()
            avg_p += p.mean().item()
            avg_k += k.mean().item()
            avg_s += s.mean().item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_train_L_data = total_L_data / len(train_loader)
        avg_train_L_mb = total_L_mb / len(train_loader)
        avg_train_L_stab = total_L_stab / len(train_loader)
        avg_aer /= len(train_loader)
        avg_p /= len(train_loader)
        avg_k /= len(train_loader)
        avg_s /= len(train_loader)
        
        scheduler.step(avg_train_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_train_loss:.4f}")
        print(f"    Params -> AER: {avg_aer:.3f}, P: {avg_p:.3f}, K: {avg_k:.3f}, S: {avg_s:.3f}")

    torch.save({
                'trained_encoder_state_dict': model.encoder.state_dict(),
                'trained_regressor_state_dict': regressor.state_dict(),
                }, os.path.join(save_path, model_file_name))
    
    print(f"\n--- Successfully saved trained model to: {os.path.join(save_path, model_file_name)} ---")

    return model, regressor
