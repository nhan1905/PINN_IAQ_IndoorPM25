import torch
import torch.nn as nn

class MassBalanceLoss(nn.Module):
    def __init__(self, dt=1.0, epsilon=0.001, lambda_mb=1.0, lambda_stab=0.5):
        super(MassBalanceLoss, self).__init__()
        self.dt = dt
        self.epsilon = epsilon
        self.lambda_mb = lambda_mb
        self.lambda_stab = lambda_stab

    def forward(self, pred_raw, targets_raw, aer, p, k, s, c_prev, c_out):
        # Prediction Loss L_data (Eq 5 - Mean Absolute Error)
        L_data = torch.mean(torch.abs(pred_raw - targets_raw))

        # Mass Balance Loss L_mb (Eq 6 & 7)
        # r(t) = {1 - dt(AER + K)}*C_in - [C_in_prev + dt*AER*P*C_out + dt*S]
        r_t = (1.0 - self.dt * (aer + k)) * pred_raw - (c_prev + self.dt * aer * p * c_out + self.dt * s)
        L_mb = torch.mean(r_t ** 2)

        # Numerical Stability Loss L_stab (Eq 8)
        # stab_term = dt(AER + K) - (1 - epsilon)
        stab_term = self.dt * (aer + k) - (1.0 - self.epsilon)
        L_stab = torch.mean((torch.relu(stab_term)) ** 2)

        # Total Loss (Eq 9)
        total_loss = L_data + self.lambda_mb * L_mb + self.lambda_stab * L_stab

        return total_loss, L_data, L_mb, L_stab
