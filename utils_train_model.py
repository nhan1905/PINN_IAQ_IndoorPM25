# Author: Nguyen Tran Nhat Minh (Revised for Dynamic Autoregressive Physics Loop)
import os
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from mass_balance_equation import *

# ==========================================
# 1. TRAIN PURE MODEL
# ==========================================
def training_pure_model(model, regressor, device, train_loader, val_loader, learning_rate, epochs, model_file_name, save_path):
    optimizer = torch.optim.Adam(list(model.parameters()) + list(regressor.parameters()), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    best_val_loss = float('inf')
    print(f"--- Start Training Pure BiLSTM (1-step) on device: {device} ---")

    for epoch in range(epochs):
        model.train()
        regressor.train()
        train_loss = 0
        
        for past_seq, targets_seq, future_feats in train_loader:
            past_seq, targets_seq, future_feats = past_seq.to(device), targets_seq.to(device), future_feats.to(device)
            optimizer.zero_grad()
            
            _, (hidden, _) = model.encoder(past_seq)
            latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            
            loss = 0
            for t in range(targets_seq.size(1)):
                pred_pm25  = regressor(latent, future_feats[:, t, :])
                target_pm25 = targets_seq[:, t, :].view(targets_seq.size(0), 1)
                loss += criterion(pred_pm25, target_pm25)
            loss = loss / targets_seq.size(1)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(regressor.parameters()), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        # --- Validation ---
        model.eval()
        regressor.eval()
        v_loss = 0
        with torch.no_grad():
            for v_past, v_target, v_feat in val_loader:
                v_past, v_target, v_feat = v_past.to(device), v_target.to(device), v_feat.to(device)
                _, (h, _) = model.encoder(v_past)
                v_lat = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)

                step_loss = 0
                for t in range(v_target.size(1)):
                    v_pred = regressor(v_lat, v_feat[:, t, :])
                    v_target_val = v_target[:, t, :].view(v_target.size(0), 1)
                    step_loss += criterion(v_pred, v_target_val).item()
                v_loss += step_loss / v_target.size(1)

        avg_train = train_loss / len(train_loader)
        avg_val = v_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        scheduler.step(avg_val)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({'trained_encoder_state_dict': model.encoder.state_dict(),
                        'trained_regressor_state_dict': regressor.state_dict()}, 
                       os.path.join(save_path, model_file_name))
            
    return model, regressor


# ==========================================
# 2. TRAIN MB MODEL 
# ==========================================
def training_mb_model(model, regressor, device, train_loader, val_loader, learning_rate, epochs, model_file_name, save_path):
    print(f"--- Start Training MB Model ---")
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': learning_rate},     
        {'params': regressor.parameters(), 'lr': learning_rate}
    ], weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = MassBalanceLoss()
    mb_equation = MassBalanceEquation(dt=5.0/60.0).to(device)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        regressor.train()
        total_epoch_train_loss = 0
        total_epoch_ldata = 0.0
        total_epoch_lphys = 0.0

        # --- TRAINING ---
        for past_seq, targets_seq, future_feats in train_loader:
            past_seq, targets_seq, future_feats = past_seq.to(device), targets_seq.to(device), future_feats.to(device)
            optimizer.zero_grad()

            _, (hidden, _) = model(past_seq)
            latent = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) 

            batch_loss = 0
            batch_ldata = 0.0
            batch_lphys = 0.0
            prev_c_indoor = past_seq[:, -1, -1].unsqueeze(1)

            for t in range(targets_seq.size(1)):
                current_context_t = future_feats[:, t, :]

                params_logit_t = regressor(context=current_context_t, prev_c_indoor=prev_c_indoor, latent=latent)

                if len(params_logit_t.shape) == 3:
                    params_logit_t = params_logit_t[:, 0, :]

                c_outdoor_t = future_feats[:, t, 0].unsqueeze(1)
                target_t = targets_seq[:, t, :]

                pred_c_indoor_raw, aer, p, k, s, c_prev, c_out = mb_equation(params_logit_t, c_outdoor_t, prev_c_indoor)

                step_loss, step_ldata, step_lphys = criterion(pred_c_indoor_raw, target_t, aer, p, k, s, c_prev, c_out)
                batch_loss += step_loss
                batch_ldata += step_ldata.item()
                batch_lphys += step_lphys.item()

                prev_c_indoor = pred_c_indoor_raw

            avg_batch_loss = batch_loss / targets_seq.size(1)
            avg_batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(regressor.parameters()), max_norm=1.0)
            optimizer.step()

            total_epoch_train_loss += avg_batch_loss.item()
            total_epoch_ldata += batch_ldata / targets_seq.size(1)
            total_epoch_lphys += batch_lphys / targets_seq.size(1)

        # --- VALIDATION --- 
        model.eval()
        regressor.eval()
        total_epoch_val_loss = 0
        v_stats = {'aer': 0, 'p': 0, 'k': 0, 's': 0}

        with torch.no_grad():
            for v_past_seq, v_targets_seq, v_future_feats in val_loader:
                v_past_seq, v_targets_seq, v_future_feats = v_past_seq.to(device), v_targets_seq.to(device), v_future_feats.to(device)

                _, (v_hidden, _) = model(v_past_seq)
                v_latent = torch.cat((v_hidden[-2, :, :], v_hidden[-1, :, :]), dim=1)

                v_prev_c_indoor = v_past_seq[:, -1, -1].unsqueeze(1)
                v_batch_loss = 0

                for t in range(v_targets_seq.size(1)):
                    v_current_context_t = v_future_feats[:, t, :]
                    v_params_logit_t = regressor(v_current_context_t, v_prev_c_indoor, v_latent)
                    
                    if len(v_params_logit_t.shape) == 3:
                        v_params_logit_t = v_params_logit_t[:, 0, :]
                    
                    v_pred, v_aer, v_p, v_k, v_s, v_cp, v_co = mb_equation(
                        v_params_logit_t, v_future_feats[:, t, 0].unsqueeze(1), v_prev_c_indoor
                    )
                    v_step_loss, _, _ = criterion(v_pred, v_targets_seq[:, t, :], v_aer, v_p, v_k, v_s, v_cp, v_co)
                    v_batch_loss += v_step_loss
                    
                    v_stats['aer'] += v_aer.mean().item()
                    v_stats['p'] += v_p.mean().item()
                    v_stats['k'] += v_k.mean().item()
                    v_stats['s'] += v_s.mean().item()
                    
                    v_prev_c_indoor = v_pred

                total_epoch_val_loss += (v_batch_loss / v_targets_seq.size(1)).item()

        final_train_loss = total_epoch_train_loss / len(train_loader)
        final_val_loss = total_epoch_val_loss / len(val_loader)
        final_ldata = total_epoch_ldata / len(train_loader)
        final_lphys = total_epoch_lphys / len(train_loader)

        total_steps_val = len(val_loader) * v_targets_seq.size(1)
        print(f"Epoch [{epoch+1}/{epochs}] | Train: {final_train_loss:.4f} (data={final_ldata:.4f} + phys={final_lphys:.4f}) | Val: {final_val_loss:.4f}")
        print(f"   >>> Stats | AER: {v_stats['aer']/total_steps_val:.4f} | P: {v_stats['p']/total_steps_val:.4f} | K: {v_stats['k']/total_steps_val:.4f} | S: {v_stats['s']/total_steps_val:.4f}")
        
        scheduler.step(final_val_loss)
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            torch.save({'trained_encoder_state_dict': model.encoder.state_dict(), 
                        'trained_regressor_state_dict': regressor.state_dict()}, 
                       os.path.join(save_path, model_file_name))
            print(f"   *** Saved best model at Epoch {epoch+1}")

        # if epoch == (epochs - 1):
        #     best_val_loss = final_val_loss
        #     torch.save({'trained_encoder_state_dict': model.encoder.state_dict(), 
        #                 'trained_regressor_state_dict': regressor.state_dict()}, 
        #                os.path.join(save_path, model_file_name))
        #     print(f"   *** Saved final model at Epoch {epoch+1}")

    return model, regressor