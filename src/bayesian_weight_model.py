import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianWeightModel(nn.Module):
    def __init__(self, feat_dim: int, weight_dim: int, device: str = "cpu"):
        super().__init__()
        self.linear = nn.Linear(feat_dim, weight_dim, bias=True)
        nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)
        self.to(device)
        self.device = device

    @torch.no_grad()
    def predict(self, feat_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = torch.tensor(feat_vec, dtype=torch.float32, device=self.device).view(1, -1)
        logits = self.linear(x)
        lam = F.softmax(logits, dim=-1).squeeze(0)
        uncertainty = np.ones_like(lam.detach().cpu().numpy())
        return lam.detach().cpu().numpy().astype(np.float32), uncertainty

    def fit(self, feats: np.ndarray, targets: np.ndarray, lr: float = 0.05):
        self.train()
        x = torch.tensor(feats, dtype=torch.float32, device=self.device)
        y = torch.tensor(targets, dtype=torch.float32, device=self.device)
        opt = torch.optim.SGD(self.parameters(), lr=lr)
        opt.zero_grad()
        logits = self.linear(x)
        pred = F.log_softmax(logits, dim=-1)
        loss = F.kl_div(pred, y, reduction="batchmean")
        loss.backward(); opt.step()
        self.eval()