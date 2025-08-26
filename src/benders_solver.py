import numpy as np
from typing import Tuple

def project_to_simplex(x: np.ndarray) -> np.ndarray:
    """Project vector onto probability simplex using efficient algorithm"""
    if np.all(x >= 0) and abs(x.sum() - 1.0) < 1e-6:
        return x
    
    u = np.sort(x)[::-1]  # Sort in descending order
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, len(x) + 1)
    cond = u - cssv / ind > 0
    
    if not np.any(cond):
        return np.ones_like(x) / float(len(x))
        
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(x - theta, 0)
    
    s = w.sum()
    return w / s if s > 0 else np.ones_like(x) / float(len(x))


def select_candidate(props: np.ndarray, metr: np.ndarray, lam: np.ndarray) -> Tuple[np.ndarray, int]:
    """Select trajectory with minimal weighted cost g(ξ,y,λ) = λ^T c(y)"""
    costs = metr @ lam
    idx = int(np.argmin(costs))
    return props[idx], idx


def cvar_eta_update(losses: np.ndarray, alpha: float, prev_eta: float = None) -> float:
    """Update VaR quantile: η = z_(⌈αM⌉) (Equation 3 in paper)"""
    if losses.size == 0:
        return prev_eta if prev_eta is not None else 0.0
    
    # Sort losses and find α-quantile
    sorted_losses = np.sort(losses)
    k = max(0, min(len(sorted_losses) - 1, int(np.ceil(alpha * len(sorted_losses)) - 1)))
    return float(sorted_losses[k])


def cvar_subgrad_lambda(losses: np.ndarray, metrics_sel: np.ndarray, eta: float, alpha: float) -> np.ndarray:
    """Compute CVaR subgradient for λ update (Section 4.4, Equation 7)"""
    if losses.size == 0 or metrics_sel.size == 0:
        return np.zeros(metrics_sel.shape[1] if len(metrics_sel.shape) > 1 else 1, dtype=np.float32)
    
    # Find exceeders: samples where Q(ξ,λ) > η
    mask = losses > eta
    if not np.any(mask):
        return np.zeros(metrics_sel.shape[1], dtype=np.float32)
    
    # CVaR subgradient: (1/(1-α)) * E[c(y*) | Q(ξ,λ) > η]
    exceeder_metrics = metrics_sel[mask]
    subgrad = exceeder_metrics.mean(axis=0) / max(1e-6, (1.0 - alpha))
    return subgrad.astype(np.float32)


def compute_cvar_value(losses: np.ndarray, alpha: float) -> float:
    """Compute CVaR value: η + (1/(1-α)) * E[(Q-η)₊]"""
    if losses.size == 0:
        return 0.0
    
    eta = cvar_eta_update(losses, alpha)
    tail_expectation = np.mean(np.maximum(losses - eta, 0))
    cvar = eta + tail_expectation / max(1e-6, (1.0 - alpha))
    return float(cvar)