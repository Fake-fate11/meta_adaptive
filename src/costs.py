import numpy as np

def _vel(xy: np.ndarray, dt: float) -> np.ndarray:
    v = np.diff(xy, axis=0, prepend=xy[0:1]) / max(1e-6, dt)
    return v

def _acc(v: np.ndarray, dt: float) -> np.ndarray:
    a = np.diff(v, axis=0, prepend=v[0:1]) / max(1e-6, dt)
    return a

def _jerk(a: np.ndarray, dt: float) -> np.ndarray:
    j = np.diff(a, axis=0, prepend=a[0:1]) / max(1e-6, dt)
    return j

def compute_metrics_vector(pred_xy: np.ndarray, gt_xy: np.ndarray, dt: float) -> np.ndarray:
    """Compute trajectory metrics with proper scaling"""
    T = min(len(pred_xy), len(gt_xy))
    if T == 0:
        return np.zeros(4, dtype=np.float32)
        
    P = pred_xy[:T]; G = gt_xy[:T]
    
    # Velocity, acceleration, jerk
    v_pred = _vel(P, dt); a_pred = _acc(v_pred, dt); j_pred = _jerk(a_pred, dt)
    
    # RMS metrics (scaled to reasonable ranges)
    jerk_rms = float(np.sqrt(np.mean(np.sum(j_pred**2, axis=1))))
    acc_rms = float(np.sqrt(np.mean(np.sum(a_pred**2, axis=1))))
    
    # Speed smoothness: RMS of speed changes
    speed_pred = np.linalg.norm(v_pred, axis=1)
    speed_changes = np.diff(speed_pred, prepend=speed_pred[0])
    smooth = float(np.sqrt(np.mean(speed_changes**2)))
    
    # Lane deviation: trajectory error relative to ground truth
    lane_dev = float(np.mean(np.linalg.norm(P - G, axis=1)))
    
    return np.array([jerk_rms, acc_rms, smooth, lane_dev], dtype=np.float32)


def compute_trajectory_diversity(trajectories: np.ndarray) -> float:
    """Compute diversity score for trajectory set"""
    if trajectories.shape[0] <= 1:
        return 0.0
    
    diversity = 0.0
    count = 0
    
    for i in range(trajectories.shape[0]):
        for j in range(i+1, trajectories.shape[0]):
            # Average endpoint distance
            endpoint_dist = np.linalg.norm(trajectories[i][-1] - trajectories[j][-1])
            # Average path distance
            path_dist = np.mean(np.linalg.norm(trajectories[i] - trajectories[j], axis=1))
            diversity += endpoint_dist + path_dist
            count += 1
    
    return diversity / max(1, count)