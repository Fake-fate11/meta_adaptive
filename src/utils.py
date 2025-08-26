import os, csv, random, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_csv_line(path: str, row: dict):
    header = list(row.keys()); exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

def plot_trajectory_comparison(gt_traj, tpp_traj, fixed_traj, adaptive_traj, save_path):
    plt.figure(figsize=(10, 8))
    
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'k-', linewidth=3, label='Ground Truth', marker='o', markersize=4)
    plt.plot(tpp_traj[:, 0], tpp_traj[:, 1], 'r--', linewidth=2, label='Uniform Weights', marker='s', markersize=3)
    plt.plot(fixed_traj[:, 0], fixed_traj[:, 1], 'b--', linewidth=2, label='Comfort-Focused', marker='^', markersize=3)
    plt.plot(adaptive_traj[:, 0], adaptive_traj[:, 1], 'g--', linewidth=2, label='Adaptive CVaR', marker='d', markersize=3)
    
    plt.scatter(gt_traj[0, 0], gt_traj[0, 1], c='black', s=100, marker='o', label='Start', zorder=5)
    plt.scatter(gt_traj[-1, 0], gt_traj[-1, 1], c='red', s=100, marker='*', label='Goal', zorder=5)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Two-Stage CVaR Trajectory Comparison')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_metrics_summary(ades, fdes, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Assuming ades/fdes have shape (n_samples, n_methods)
    n_methods = ades.shape[1] // 2  # Each method contributes 2 columns (ADE, FDE)
    methods = ['Uniform', 'Comfort', 'Adaptive'][:n_methods]
    
    ade_means = [ades[:, i*2].mean() for i in range(n_methods)]
    ade_stds = [ades[:, i*2].std() for i in range(n_methods)]
    fde_means = [fdes[:, i*2+1].mean() for i in range(n_methods)]
    fde_stds = [fdes[:, i*2+1].std() for i in range(n_methods)]
    
    x_pos = np.arange(len(methods))
    colors = ['red', 'blue', 'green'][:len(methods)]
    
    ax1.bar(x_pos, ade_means, yerr=ade_stds, capsize=5, color=colors, alpha=0.7)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('ADE (m)')
    ax1.set_title('Average Displacement Error')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods)
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(x_pos, fde_means, yerr=fde_stds, capsize=5, color=colors, alpha=0.7)
    ax2.set_xlabel('Method')
    ax2.set_ylabel('FDE (m)')
    ax2.set_title('Final Displacement Error')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()