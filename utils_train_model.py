# Author: Nguyen Tran Nhat Minh (Revised for Dynamic Autoregressive Physics Loop)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from mass_balance_equation import *


def _calculate_pm25_slopes(current_window, long_lag=5):
    if current_window.size(1) < 2:
        raise ValueError(
            "At least two time steps are required"
        )

    lag = min(long_lag, current_window.size(1))
    latest_pm25 = current_window[:, -1, -1]

    slope_1 = (
        latest_pm25
        - current_window[:, -2, -1]
    ).unsqueeze(1)

    slope_5 = (
        (
            latest_pm25
            - current_window[:, -lag, -1]
        )
        / float(lag - 1)
    ).unsqueeze(1)

    return slope_1, slope_5

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
            
            _, (hidden, _) = model(past_seq)
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
                _, (h, _) = model(v_past)
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
            torch.save({'trained_model_state_dict': model.state_dict(),
                        'trained_regressor_state_dict': regressor.state_dict()}, 
                       os.path.join(save_path, model_file_name))
            
    return model, regressor


# ==========================================
# 2. TRAIN MB MODEL 
# ==========================================
def training_mb_model(
    model,
    regressor,
    device,
    train_loader,
    val_loader,
    learning_rate,
    epochs,
    model_file_name,
    save_path,
    lambda_teacher=0.2,
    lambda_trend=0.1,
    lambda_bias=0.5,
    high_pm_threshold=50.0,
    high_pm_weight=1.0,
    horizon_max_weight=2.0,
    early_stopping_patience=None
):
    print(f"--- Start Training MB Model ---")

    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1")

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": learning_rate},
            {"params": regressor.parameters(), "lr": learning_rate}
        ],
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15
    )

    criterion = MassBalanceLoss()
    mb_equation = MassBalanceEquation(dt=1.0 / 60.0).to(device)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        # =====================================================
        # TRAINING
        # =====================================================
        model.train()
        regressor.train()

        total_train_loss = 0.0
        total_train_data_loss = 0.0
        total_train_physics_loss = 0.0
        total_train_teacher_loss = 0.0
        total_train_trend_loss = 0.0
        total_train_bias_loss = 0.0
        total_train_samples = 0

        for past_seq, targets_seq, future_feats in train_loader:
            past_seq = past_seq.to(device)
            targets_seq = targets_seq.to(device)
            future_feats = future_feats.to(device)

            optimizer.zero_grad()

            batch_size = past_seq.size(0)
            prediction_steps = targets_seq.size(1)

            current_window = past_seq

            batch_loss = 0.0
            batch_data_loss = 0.0
            batch_physics_loss = 0.0
            batch_teacher_loss = 0.0
            batch_trend_loss = 0.0
            step_predictions = []
            horizon_weight_sum = 0.0

            for t in range(prediction_steps):
                # Recalculate latent using the latest window
                _, (hidden, _) = model(current_window)

                latent = torch.cat(
                    (hidden[-2, :, :], hidden[-1, :, :]),
                    dim=1
                )

                prev_c_indoor = current_window[:, -1, -1].unsqueeze(1)
                current_context = future_feats[:, t, :]
                slope_1, slope_5 = _calculate_pm25_slopes(current_window)

                params_logit = regressor(
                    context=current_context,
                    prev_c_indoor=prev_c_indoor,
                    slope_1=slope_1,
                    slope_5=slope_5,
                    latent=latent
                )

                if params_logit.ndim == 3:
                    params_logit = params_logit[:, 0, :]

                c_outdoor = current_context[:, 0:1]
                target = targets_seq[:, t, :].reshape(batch_size, 1)

                if t == 0:
                    actual_prev_c_indoor = (
                        past_seq[:, -1, -1].unsqueeze(1)
                    )
                else:
                    actual_prev_c_indoor = targets_seq[
                        :, t - 1, :
                    ].reshape(batch_size, 1)

                (
                    pred_c_indoor,
                    aer,
                    p,
                    k,
                    s,
                    c_prev,
                    c_out
                ) = mb_equation(
                    params_logit,
                    c_outdoor,
                    prev_c_indoor
                )

                (
                    _,
                    _,
                    step_physics_loss
                ) = criterion(
                    pred_c_indoor,
                    target,
                    aer,
                    p,
                    k,
                    s,
                    c_prev,
                    c_out
                )

                teacher_prediction, _, _, _, _, _, _ = mb_equation(
                    params_logit,
                    c_outdoor,
                    actual_prev_c_indoor
                )

                teacher_loss = F.smooth_l1_loss(
                    teacher_prediction,
                    target
                )

                predicted_change = pred_c_indoor - prev_c_indoor
                actual_change = target - actual_prev_c_indoor
                trend_loss = F.smooth_l1_loss(
                    predicted_change,
                    actual_change
                )

                # High PM2.5 values are rare in the training set. Give them
                # more influence without allowing a few peaks to dominate.
                target_weight = 1.0 + high_pm_weight * torch.clamp(
                    (target - high_pm_threshold) / high_pm_threshold,
                    min=0.0,
                    max=2.0
                )
                weighted_data_loss = torch.mean(
                    target_weight * (pred_c_indoor - target) ** 2
                )

                step_loss = (
                    weighted_data_loss
                    + step_physics_loss
                    + lambda_teacher * teacher_loss
                    + lambda_trend * trend_loss
                )

                if prediction_steps > 1:
                    horizon_weight = 1.0 + (
                        horizon_max_weight - 1.0
                    ) * t / float(prediction_steps - 1)
                else:
                    horizon_weight = 1.0

                batch_loss = batch_loss + horizon_weight * step_loss
                horizon_weight_sum += horizon_weight
                batch_data_loss += horizon_weight * weighted_data_loss.item()
                batch_physics_loss += horizon_weight * step_physics_loss.item()
                batch_teacher_loss += horizon_weight * teacher_loss.item()
                batch_trend_loss += horizon_weight * trend_loss.item()
                step_predictions.append(pred_c_indoor)

                # Insert prediction into the window for the next step
                new_row = torch.cat(
                    (current_context, pred_c_indoor),
                    dim=1
                ).unsqueeze(1)

                current_window = torch.cat(
                    (current_window[:, 1:, :], new_row),
                    dim=1
                )

            pred_sequence = torch.stack(step_predictions, dim=1)
            bias_loss = torch.mean(
                torch.mean(pred_sequence - targets_seq, dim=1) ** 2
            )

            average_batch_loss = (
                batch_loss / horizon_weight_sum
                + lambda_bias * bias_loss
            )
            average_batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(regressor.parameters()),
                max_norm=1.0
            )

            optimizer.step()

            total_train_loss += average_batch_loss.item() * batch_size
            total_train_data_loss += (
                batch_data_loss / horizon_weight_sum
            ) * batch_size
            total_train_physics_loss += (
                batch_physics_loss / horizon_weight_sum
            ) * batch_size
            total_train_teacher_loss += (
                batch_teacher_loss / horizon_weight_sum
            ) * batch_size
            total_train_trend_loss += (
                batch_trend_loss / horizon_weight_sum
            ) * batch_size
            total_train_bias_loss += bias_loss.item() * batch_size
            total_train_samples += batch_size

        epoch_train_loss = total_train_loss / total_train_samples
        epoch_data_loss = total_train_data_loss / total_train_samples
        epoch_physics_loss = (
            total_train_physics_loss / total_train_samples
        )
        epoch_teacher_loss = (
            total_train_teacher_loss / total_train_samples
        )
        epoch_trend_loss = (
            total_train_trend_loss / total_train_samples
        )
        epoch_bias_loss = (
            total_train_bias_loss / total_train_samples
        )

        # =====================================================
        # VALIDATION
        # =====================================================
        model.eval()
        regressor.eval()

        total_val_loss = 0.0
        total_val_bias_loss = 0.0
        total_val_samples = 0

        val_parameter_sums = {
            "aer": 0.0,
            "p": 0.0,
            "k": 0.0,
            "s": 0.0
        }
        total_parameter_values = 0

        with torch.no_grad():
            for (
                val_past_seq,
                val_targets_seq,
                val_future_feats
            ) in val_loader:
                val_past_seq = val_past_seq.to(device)
                val_targets_seq = val_targets_seq.to(device)
                val_future_feats = val_future_feats.to(device)

                batch_size = val_past_seq.size(0)
                prediction_steps = val_targets_seq.size(1)

                current_window = val_past_seq
                batch_val_loss = 0.0
                val_step_predictions = []
                horizon_weight_sum = 0.0

                for t in range(prediction_steps):
                    # Recalculate latent from the updated window
                    _, (hidden, _) = model(current_window)

                    latent = torch.cat(
                        (hidden[-2, :, :], hidden[-1, :, :]),
                        dim=1
                    )

                    prev_c_indoor = (
                        current_window[:, -1, -1].unsqueeze(1)
                    )
                    current_context = val_future_feats[:, t, :]
                    slope_1, slope_5 = _calculate_pm25_slopes(
                        current_window
                    )

                    params_logit = regressor(
                        context=current_context,
                        prev_c_indoor=prev_c_indoor,
                        slope_1=slope_1,
                        slope_5=slope_5,
                        latent=latent
                    )

                    if params_logit.ndim == 3:
                        params_logit = params_logit[:, 0, :]

                    c_outdoor = current_context[:, 0:1]
                    target = val_targets_seq[:, t, :].reshape(
                        batch_size,
                        1
                    )

                    (
                        prediction,
                        aer,
                        p,
                        k,
                        s,
                        c_prev,
                        c_out
                    ) = mb_equation(
                        params_logit,
                        c_outdoor,
                        prev_c_indoor
                    )

                    _, _, step_physics_loss = criterion(
                        prediction,
                        target,
                        aer,
                        p,
                        k,
                        s,
                        c_prev,
                        c_out
                    )

                    target_weight = 1.0 + high_pm_weight * torch.clamp(
                        (target - high_pm_threshold) / high_pm_threshold,
                        min=0.0,
                        max=2.0
                    )
                    weighted_data_loss = torch.mean(
                        target_weight * (prediction - target) ** 2
                    )
                    step_loss = weighted_data_loss + step_physics_loss

                    if prediction_steps > 1:
                        horizon_weight = 1.0 + (
                            horizon_max_weight - 1.0
                        ) * t / float(prediction_steps - 1)
                    else:
                        horizon_weight = 1.0

                    batch_val_loss += horizon_weight * step_loss
                    horizon_weight_sum += horizon_weight
                    val_step_predictions.append(prediction)

                    val_parameter_sums["aer"] += aer.sum().item()
                    val_parameter_sums["p"] += p.sum().item()
                    val_parameter_sums["k"] += k.sum().item()
                    val_parameter_sums["s"] += s.sum().item()
                    total_parameter_values += aer.numel()

                    # Autoregressive window update
                    new_row = torch.cat(
                        (current_context, prediction),
                        dim=1
                    ).unsqueeze(1)

                    current_window = torch.cat(
                        (current_window[:, 1:, :], new_row),
                        dim=1
                    )

                val_pred_sequence = torch.stack(
                    val_step_predictions,
                    dim=1
                )
                val_bias_loss = torch.mean(
                    torch.mean(
                        val_pred_sequence - val_targets_seq,
                        dim=1
                    ) ** 2
                )

                average_batch_val_loss = (
                    batch_val_loss / horizon_weight_sum
                    + lambda_bias * val_bias_loss
                )

                total_val_loss += (
                    average_batch_val_loss.item() * batch_size
                )
                total_val_bias_loss += val_bias_loss.item() * batch_size
                total_val_samples += batch_size

        epoch_val_loss = total_val_loss / total_val_samples
        epoch_val_bias_loss = total_val_bias_loss / total_val_samples

        mean_aer = (
            val_parameter_sums["aer"] / total_parameter_values
        )
        mean_p = val_parameter_sums["p"] / total_parameter_values
        mean_k = val_parameter_sums["k"] / total_parameter_values
        mean_s = val_parameter_sums["s"] / total_parameter_values

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train: {epoch_train_loss:.4f} "
            f"(data={epoch_data_loss:.4f}, "
            f"physics={epoch_physics_loss:.4f}, "
            f"teacher={epoch_teacher_loss:.4f}, "
            f"trend={epoch_trend_loss:.4f}, "
            f"bias={epoch_bias_loss:.4f}) | "
            f"Val: {epoch_val_loss:.4f} "
            f"(bias={epoch_val_bias_loss:.4f})"
        )

        print(
            f"   >>> Stats | "
            f"AER: {mean_aer:.4f} | "
            f"P: {mean_p:.4f} | "
            f"K: {mean_k:.4f} | "
            f"S: {mean_s:.4f}"
        )

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss

            checkpoint = {
                "trained_model_state_dict": model.state_dict(),
                "trained_regressor_state_dict": regressor.state_dict(),
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "lambda_teacher": lambda_teacher,
                "lambda_trend": lambda_trend,
                "lambda_bias": lambda_bias,
                "high_pm_threshold": high_pm_threshold,
                "high_pm_weight": high_pm_weight,
                "horizon_max_weight": horizon_max_weight
            }

            torch.save(
                checkpoint,
                os.path.join(save_path, model_file_name)
            )

            print(
                f"   *** Saved best model at epoch {epoch + 1}"
            )

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    "   *** Early stopping: validation loss did not improve "
                    f"for {early_stopping_patience} epochs"
                )
                break

    return model, regressor
