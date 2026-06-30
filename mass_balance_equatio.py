import torch
import torch.nn as nn

class MassBalanceLoss(nn.Module):
    def __init__(self, dt=1.0/60.0, lambda_physics=5.0):
        super(MassBalanceLoss, self).__init__()
        self.dt = dt
        self.lambda_physics = lambda_physics 
        self.AER_soft_min = 0.20
        self.K_soft_min   = 0.10

    def forward(self, pred_seq, targets_seq, aer_seq, p_seq, k_seq, s_seq, cprev_seq, cout_seq):
        L_data = torch.mean((pred_seq - targets_seq) ** 2)

        L_aer = torch.mean(torch.relu(self.AER_soft_min - aer_seq) ** 2) / (self.AER_soft_min ** 2)
        L_k   = torch.mean(torch.relu(self.K_soft_min   - k_seq  ) ** 2) / (self.K_soft_min   ** 2)

        L_physics = self.lambda_physics * (L_aer + L_k)
        return L_data + L_physics, L_data, L_physics

class MassBalanceEquation(nn.Module):
    def __init__(self, dt=1.0/60.0):
        super(MassBalanceEquation, self).__init__()
        self.dt = dt

    def forward(self, params_logit, c_outdoor, prev_c_indoor):
        AERt = 1.45 * torch.sigmoid(params_logit[:, 0:1]) + 0.05
        Pt   = 1.00 * torch.sigmoid(params_logit[:, 1:2])          
        Kt   = 1.00 * torch.sigmoid(params_logit[:, 2:3]) + 0.05
        St   = 100.00 * torch.sigmoid(params_logit[:, 3:4]) - 1.00

        Cinprev = prev_c_indoor.to(dtype=torch.float32)
        Cout    = c_outdoor.to(dtype=torch.float32)

        numerator   = Cinprev + self.dt * (AERt * Pt * Cout + St)
        denominator = 1.0 + self.dt * (AERt + Kt)
        pred_raw    = numerator / denominator

        return pred_raw, AERt, Pt, Kt, St, Cinprev, Cout
