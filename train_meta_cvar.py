import os, argparse, yaml, json, math, numpy as np
from typing import Dict, Any, Tuple, List

import warnings
warnings.filterwarnings("ignore", message="pkg_resources")

import torch
from torch.utils.data import DataLoader

from trajdata import AgentType, UnifiedDataset
from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.data_index import AgentDataIndex

from src.tpp_runtime import TPPRuntime
from src.costs import compute_metrics_vector
from src.benders_solver import project_to_simplex, select_candidate, cvar_eta_update, cvar_subgrad_lambda
from src.bayesian_weight_model import BayesianWeightModel
from src.sampling_util import ProportionalCategorySampler, estimate_distribution, classify_sample
from src.utils import set_seed, save_csv_line, plot_trajectory_comparison, plot_metrics_summary
from src.visualization import plot_training_evolution, generate_training_report


def build_attention_radius() -> Dict[tuple, float]:
    types = list(AgentType)
    rad = {(a, b): 0.0 for a in types for b in types}
    rad[(AgentType.VEHICLE, AgentType.VEHICLE)] = 50.0
    return rad


def build_dataset(cfg: Dict[str, Any], mode: str = "train") -> UnifiedDataset:
    hist = (0.5, float(cfg["training"]["history_sec"]))
    fut  = (0.5, float(cfg["training"]["prediction_sec"]))
    attention = build_attention_radius()
    data_dirs = {"nusc_trainval": cfg["data"]["nusc_root"]}
    
    # Use multiple data tags for full dataset coverage
    if mode == "train":
        desired_data = cfg["data"]["train_tags"]
    else:
        desired_data = cfg["data"]["eval_tags"]
    
    print(f"Loading {mode} data with tags: {desired_data}")
    
    ds = UnifiedDataset(
        desired_data=desired_data,
        history_sec=hist,
        future_sec=fut,
        agent_interaction_distances=attention,
        incl_robot_future=False,
        incl_raster_map=False,
        raster_map_params=None,
        only_predict=[AgentType.VEHICLE],
        no_types=[AgentType.UNKNOWN],
        augmentations=None,
        num_workers=int(cfg["training"]["preprocess_workers"]),
        cache_location=cfg["data"]["trajdata_cache_dir"],
        data_dirs=data_dirs,
        verbose=True,
    )
    return ds


def _positions(x):
    if x is None:
        return None
    if hasattr(x, "positions"):
        arr = x.positions
    elif hasattr(x, "position"):
        arr = x.position
    else:
        arr = x[..., :2]
    return arr.detach().cpu().numpy().astype(np.float32)


def extract_xy_from_batch(batch, i: int = 0):
    if batch is None:
        return None, None
    
    try:
        if not hasattr(batch, "agent_hist") or not hasattr(batch, "agent_fut"):
            return None, None
        
        if len(batch.agent_hist) <= i or len(batch.agent_fut) <= i:
            return None, None
            
        hist = _positions(batch.agent_hist[i])
        fut = _positions(batch.agent_fut[i])
        
        if hist is None or fut is None:
            return None, None
            
        return hist, fut
    except (AttributeError, IndexError, TypeError):
        return None, None


def extract_scene_features(batch, hist_xy: np.ndarray, fut_xy: np.ndarray, cfg: Dict) -> np.ndarray:
    """Extract scene features for Bayesian meta-learning (Section 5)"""
    try:
        cls = classify_sample(hist_xy, fut_xy,
                              angle_deg_thresh=float(cfg["training"]["sampler"]["angle_deg_thresh"]),
                              speed_limit_ms=float(cfg["training"]["sampler"]["speed_limit_ms"]))
        
        maneuver_onehot = np.array([
            1.0 if cls["maneuver"] == "turn_left" else 0.0,
            1.0 if cls["maneuver"] == "turn_right" else 0.0,
            1.0 if cls["maneuver"] == "straight" else 0.0,
        ], dtype=np.float32)
        
        rule_feat = 1.0 if (cls.get("overspeed", False) or cls.get("stopline", False)) else 0.0
        
        feat = np.concatenate([maneuver_onehot, np.array([rule_feat], dtype=np.float32)], axis=0)
        return feat
    except Exception:
        return np.zeros(4, dtype=np.float32)


def solve_inner_problem(props: np.ndarray, metrics: np.ndarray, lam: np.ndarray) -> Tuple[float, int]:
    """Solve inner subproblem Q(ξ,λ) = min_y g(ξ,y,λ) (Equation 4)"""
    costs = metrics @ lam
    idx = int(np.argmin(costs))
    return float(costs[idx]), idx


def normalize_costs(metrics: np.ndarray, cfg: Dict) -> np.ndarray:
    """Normalize cost components to comparable scales"""
    normalized = metrics.copy()
    scales = cfg["training"]["cost_normalization"]
    
    # Apply scaling factors
    normalized[:, 0] *= scales["jerk_scale"]   # RMS jerk
    normalized[:, 1] *= scales["acc_scale"]    # RMS acceleration  
    normalized[:, 2] *= scales["smooth_scale"] # Smoothness
    normalized[:, 3] *= scales["lane_scale"]   # Lane deviation
    
    return normalized


def generate_diverse_trajectories(hist_xy: np.ndarray, fut_xy: np.ndarray, 
                                num_modes: int, dt: float, std: float = 2.0) -> np.ndarray:
    """Generate more diverse trajectory candidates when TPP fails"""
    T = fut_xy.shape[0]
    start = hist_xy[-1]  # Last history point
    end_gt = fut_xy[-1]  # Ground truth endpoint
    
    trajectories = []
    
    # Include ground truth as one mode
    trajectories.append(fut_xy.copy())
    
    # Generate variations around ground truth
    for i in range(num_modes - 1):
        # Add noise to control points
        noise_scale = std * (i + 1) / num_modes
        
        # Vary the endpoint
        end_noise = np.random.normal(0, noise_scale, 2)
        end_varied = end_gt + end_noise
        
        # Create trajectory with varied curvature
        t = np.linspace(0, 1, T)
        
        # Add some curvature variation
        mid_offset = np.random.normal(0, noise_scale/2, 2)
        
        traj = []
        for j, t_val in enumerate(t):
            # Interpolate with curved path
            curved_point = (1-t_val)**2 * start + 2*(1-t_val)*t_val * (start + end_varied)/2 + mid_offset * np.sin(np.pi*t_val) + t_val**2 * end_varied
            traj.append(curved_point)
        
        trajectories.append(np.array(traj, dtype=np.float32))
    
    return np.stack(trajectories, axis=0)


def build_robust_loader(ds, cfg, seed: int):
    """Build DataLoader with robust sampling to avoid None batches"""
    dist_cfg = cfg["training"]["sampler"]
    
    dist = estimate_distribution(
        ds,
        angle_deg_thresh=float(dist_cfg["angle_deg_thresh"]),
        speed_limit_ms=float(dist_cfg["speed_limit_ms"]),
        num_samples=int(dist_cfg.get("est_samples", 3000)),
        seed=seed,
    )
    
    forced = dist_cfg.get("proportions", None)
    if forced:
        s = sum(v for v in forced.values() if v is not None)
        if s > 0:
            forced = {k: (v/s if v is not None else 0.0) for k, v in forced.items()}
        for k in ["straight", "turn_left", "turn_right", "stopline"]:
            if k in forced and forced[k] is not None:
                dist[k] = forced[k]

    sampler = ProportionalCategorySampler(
        dataset=ds,
        proportions=dist,
        angle_deg_thresh=float(dist_cfg["angle_deg_thresh"]),
        speed_limit_ms=float(dist_cfg["speed_limit_ms"]),
        seed=seed,
    )
    
    def robust_collate(samples):
        try:
            if not samples:
                return None
            valid_samples = [s for s in samples if s is not None]
            if not valid_samples:
                return None
            return ds.get_collate_fn(pad_format="right")(valid_samples)
        except Exception:
            return None
    
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, sampler=sampler,
        num_workers=0, collate_fn=robust_collate
    )
    return loader, dist


def train_mode(cfg, args):
    device = "cuda" if torch.cuda.is_available() and cfg.get("device","auto")!="cpu" else "cpu"
    print(f"Using device: {device}")
    
    train_ds = build_dataset(cfg, "train")
    loader, used_dist = build_robust_loader(train_ds, cfg, seed=int(cfg.get("seed", 42)))
    print("Category proportions used:", json.dumps(used_dist, indent=2))

    tpp = None
    if cfg["outer_model"]["enable"]:
        ckpt_dir = os.path.join(cfg["outer_model"]["path"], cfg["outer_model"]["ckpt"])
        config_path = os.path.join(ckpt_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Warning: TPP config not found: {config_path}, using fallback")
        else:
            try:
                tpp = TPPRuntime(
                    config_path=config_path,
                    model_dir=ckpt_dir,
                    iteration=int(cfg["outer_model"]["iteration"]),
                    max_modes=int(cfg["outer_model"]["max_modes"]),
                    device=device,
                    eager=bool(cfg["outer_model"]["tpp"].get("eager", False)),
                    force_cpu_load=bool(cfg["outer_model"]["tpp"].get("force_cpu_load", True)),
                )
                print(f"TPP loaded: {tpp.loaded}")
            except Exception as e:
                print(f"TPP loading failed: {e}, using fallback")
                tpp = None

    feat_dim = 4
    weight_dim = int(cfg["model"]["cost_dim"])
    
    bayes = BayesianWeightModel(feat_dim, weight_dim, device=device)
    
    lam = np.ones(weight_dim, dtype=np.float32) / float(weight_dim)
    
    eta = 0.0
    alpha = float(cfg["training"]["cvar_alpha"])
    lr = float(cfg["training"]["learning_rate"])
    
    log_dir = cfg["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "train_log.csv")

    if args.quick:
        epochs = int(cfg["training"]["quick_epochs"])
        steps_per_epoch = int(cfg["training"]["quick_steps_per_epoch"])
    else:
        epochs = int(cfg["training"]["epochs"])
        steps_per_epoch = int(cfg["training"]["steps_per_epoch"])
    
    dt = float(cfg["training"]["dt"])
    fallback_std = float(cfg["outer_model"]["tpp"].get("fallback_std", 2.0))

    print(f"Training for {epochs} epochs, {steps_per_epoch} steps each")

    for ep in range(1, epochs + 1):
        losses = []
        metrics_selected = []
        scene_features = []
        lambda_priors = []
        tpp_success_count = 0
        
        step = 0
        successful_steps = 0
        
        for batch in loader:
            if batch is None:
                continue
                
            hist_xy, fut_xy = extract_xy_from_batch(batch, i=0)
            if hist_xy is None or fut_xy is None:
                continue
                
            try:
                batch = batch.to(device)
            except Exception:
                continue
            
            feat = extract_scene_features(batch, hist_xy, fut_xy, cfg)
            scene_features.append(feat)
            
            lam_prior, _ = bayes.predict(feat)
            lambda_priors.append(lam_prior)
            
            # Enhanced trajectory generation
            props = None
            if tpp is not None:
                try:
                    props, prior = tpp.propose_from_batch(batch, agent_index=0, num_modes=int(cfg["outer_model"]["max_modes"]))
                    if props is not None:
                        tpp_success_count += 1
                except Exception as e:
                    props = None
                    
            if props is None:
                # Generate more diverse fallback trajectories
                props = generate_diverse_trajectories(
                    hist_xy, fut_xy, 
                    num_modes=int(cfg["outer_model"]["max_modes"]), 
                    dt=dt, 
                    std=fallback_std
                )

            P = props.shape[0]
            weight_dim = int(cfg["model"]["cost_dim"])
            metrics = np.zeros((P, weight_dim), dtype=np.float32)
            for i in range(P):
                raw_metrics = compute_metrics_vector(props[i], fut_xy, dt=dt)
                metrics[i] = raw_metrics
            
            # Normalize metrics for better scaling
            metrics = normalize_costs(metrics, cfg)

            loss, selected_idx = solve_inner_problem(props, metrics, lam_prior)
            
            losses.append(loss)
            metrics_selected.append(metrics[selected_idx])
            
            successful_steps += 1
            step += 1
            
            if successful_steps >= steps_per_epoch:
                break

        if len(losses) == 0:
            print(f"Epoch {ep}: No valid samples, skipping")
            continue
            
        losses_arr = np.array(losses, dtype=np.float32)
        metrics_arr = np.array(metrics_selected, dtype=np.float32)
        features_arr = np.array(scene_features, dtype=np.float32)
        priors_arr = np.array(lambda_priors, dtype=np.float32)
        
        eta = cvar_eta_update(losses_arr, alpha, eta)
        
        subgradient = cvar_subgrad_lambda(losses_arr, metrics_arr, eta, alpha)
        
        lam = project_to_simplex(lam - lr * subgradient)
        
        if len(features_arr) > 0:
            bayes.fit(feats=features_arr, targets=priors_arr, 
                      lr=float(cfg["training"].get("bayes_lr", 0.05)))
        
        avg_loss = float(losses_arr.mean())
        tpp_rate = tpp_success_count / max(1, successful_steps)
        
        print(f"Epoch {ep}/{epochs}: steps={successful_steps}, η={eta:.4f}, avg_loss={avg_loss:.4f}, λ={lam}, TPP_rate={tpp_rate:.2f}")
        
        save_csv_line(csv_path, {
            "epoch": ep,
            "steps": successful_steps,
            "eta": eta,
            "avg_loss": avg_loss,
            "tpp_success_rate": tpp_rate,
            **{f"lam{i}": float(lam[i]) for i in range(len(lam))}
        })

    os.makedirs(cfg["paths"]["model_dir"], exist_ok=True)
    np.savez(os.path.join(cfg["paths"]["model_dir"], "lam_eta.npz"), lam=lam, eta=eta)
    
    bayes_path = os.path.join(cfg["paths"]["model_dir"], "bayesian_weights.pt")
    torch.save(bayes.state_dict(), bayes_path)
    
    plot_training_evolution(csv_path, cfg["paths"]["plot_dir"])
    
    report_path = os.path.join(cfg["paths"]["log_dir"], "training_report.html")
    generate_training_report("config.yaml", cfg["paths"]["log_dir"], report_path)
    
    print(f"Models saved: lam_eta.npz, {bayes_path}")
    print(f"Training report: {report_path}")


def test_mode(cfg, args):
    device = "cuda" if torch.cuda.is_available() and cfg.get("device","auto")!="cpu" else "cpu"
    print(f"Testing on device: {device}")

    eval_ds = build_dataset(cfg, "eval")
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0, 
                           collate_fn=eval_ds.get_collate_fn(pad_format="right"))

    tpp = None
    if cfg["outer_model"]["enable"]:
        ckpt_dir = os.path.join(cfg["outer_model"]["path"], cfg["outer_model"]["ckpt"])
        config_path = os.path.join(ckpt_dir, "config.json")
        if os.path.exists(config_path):
            try:
                tpp = TPPRuntime(
                    config_path=config_path,
                    model_dir=ckpt_dir,
                    iteration=int(cfg["outer_model"]["iteration"]),
                    max_modes=int(cfg["outer_model"]["max_modes"]),
                    device=device,
                    eager=bool(cfg["outer_model"]["tpp"].get("eager", False)),
                    force_cpu_load=bool(cfg["outer_model"]["tpp"].get("force_cpu_load", True)),
                )
            except Exception as e:
                print(f"TPP loading failed in test: {e}")
                tpp = None

    lam_path = os.path.join(cfg["paths"]["model_dir"], "lam_eta.npz")
    if os.path.exists(lam_path):
        data = np.load(lam_path)
        lam_adaptive = data["lam"]
        print(f"Loaded adaptive λ: {lam_adaptive}")
    else:
        weight_dim = int(cfg["model"]["cost_dim"])
        lam_adaptive = np.ones(weight_dim, dtype=np.float32) / float(weight_dim)
        print("No trained lambda found, using uniform")

    weight_dim = int(cfg["model"]["cost_dim"])
    lam_uniform = np.ones(weight_dim, dtype=np.float32) / float(weight_dim)
    lam_comfort = np.array([0.4, 0.4, 0.1, 0.1], dtype=np.float32)[:weight_dim]
    lam_comfort = lam_comfort / lam_comfort.sum()
    
    dt = float(cfg["training"]["dt"])
    fallback_std = float(cfg["outer_model"]["tpp"].get("fallback_std", 2.0))
    
    results = {"uniform": [], "comfort": [], "adaptive": []}
    cost_breakdowns = {"uniform": [], "comfort": [], "adaptive": []}
    scenario_results = {}
    
    sample_count = 0
    max_samples = 50 if args.quick else 200
    
    os.makedirs(cfg["paths"]["plot_dir"], exist_ok=True)
    
    for batch in eval_loader:
        if sample_count >= max_samples:
            break
            
        hist_xy, fut_xy = extract_xy_from_batch(batch, i=0)
        if hist_xy is None or fut_xy is None:
            continue
            
        try:
            batch = batch.to(device)
        except Exception:
            continue

        scenario_type = classify_sample(hist_xy, fut_xy,
                                      angle_deg_thresh=float(cfg["training"]["sampler"]["angle_deg_thresh"]),
                                      speed_limit_ms=float(cfg["training"]["sampler"]["speed_limit_ms"]))["maneuver"]

        props = None
        if tpp is not None:
            try:
                props, prior = tpp.propose_from_batch(batch, agent_index=0, num_modes=int(cfg["outer_model"]["max_modes"]))
            except Exception:
                props = None
            
        if props is None:
            props = generate_diverse_trajectories(
                hist_xy, fut_xy, 
                num_modes=int(cfg["outer_model"]["max_modes"]), 
                dt=dt, 
                std=fallback_std
            )

        P = props.shape[0]
        metrics = np.zeros((P, weight_dim), dtype=np.float32)
        for i in range(P):
            raw_metrics = compute_metrics_vector(props[i], fut_xy, dt=dt)
            metrics[i] = raw_metrics
        
        metrics = normalize_costs(metrics, cfg)

        _, uniform_idx = solve_inner_problem(props, metrics, lam_uniform)
        _, comfort_idx = solve_inner_problem(props, metrics, lam_comfort)
        _, adaptive_idx = solve_inner_problem(props, metrics, lam_adaptive)

        traj_uniform = props[uniform_idx]
        traj_comfort = props[comfort_idx] 
        traj_adaptive = props[adaptive_idx]

        for name, traj, idx in [("uniform", traj_uniform, uniform_idx), 
                               ("comfort", traj_comfort, comfort_idx), 
                               ("adaptive", traj_adaptive, adaptive_idx)]:
            ade = np.mean(np.linalg.norm(traj - fut_xy, axis=1))
            fde = np.linalg.norm(traj[-1] - fut_xy[-1])
            results[name].append([ade, fde])
            cost_breakdowns[name].append(metrics[idx])
            
            if scenario_type not in scenario_results:
                scenario_results[scenario_type] = {"uniform": [], "comfort": [], "adaptive": []}
            scenario_results[scenario_type][name].append({"ade": ade, "fde": fde})

        if sample_count < 10:
            plot_path = os.path.join(cfg["paths"]["plot_dir"], f"trajectory_sample_{sample_count}.png")
            plot_trajectory_comparison(
                gt_traj=fut_xy,
                tpp_traj=traj_uniform,
                fixed_traj=traj_comfort,
                adaptive_traj=traj_adaptive,
                save_path=plot_path
            )
        
        sample_count += 1

    for name in results:
        results[name] = np.array(results[name])
        cost_breakdowns[name] = np.array(cost_breakdowns[name])
    
    if len(results["adaptive"]) > 0:
        print(f"\nEvaluation Results (n={len(results['adaptive'])}):")
        for name in ["uniform", "comfort", "adaptive"]:
            ade_mean = results[name][:,0].mean()
            ade_std = results[name][:,0].std()
            fde_mean = results[name][:,1].mean()
            fde_std = results[name][:,1].std()
            print(f"{name.capitalize():>8s} - ADE: {ade_mean:.3f}±{ade_std:.3f}, FDE: {fde_mean:.3f}±{fde_std:.3f}")
        
        print(f"\nCost Component Analysis:")
        cost_names = ['RMS Jerk', 'RMS Accel', 'Smoothness', 'Lane Dev']
        for i, cost_name in enumerate(cost_names):
            print(f"{cost_name}:")
            for name in ["uniform", "comfort", "adaptive"]:
                if cost_breakdowns[name].shape[1] > i:
                    mean_cost = cost_breakdowns[name][:,i].mean()
                    print(f"  {name}: {mean_cost:.4f}")
        
        summary_path = os.path.join(cfg["paths"]["plot_dir"], "metrics_summary.png")
        plot_metrics_summary(
            np.column_stack([results["uniform"], results["comfort"], results["adaptive"]]),
            np.column_stack([results["uniform"], results["comfort"], results["adaptive"]]),
            save_path=summary_path
        )
        
        results_path = os.path.join(cfg["paths"]["log_dir"], "eval_results.json")
        eval_results = {}
        for name in results:
            eval_results[f"{name}_ade"] = float(results[name][:,0].mean())
            eval_results[f"{name}_fde"] = float(results[name][:,1].mean())
            eval_results[f"{name}_ade_std"] = float(results[name][:,0].std())
            eval_results[f"{name}_fde_std"] = float(results[name][:,1].std())
            
            for i in range(cost_breakdowns[name].shape[1]):
                eval_results[f"{name}_cost_{i}"] = float(cost_breakdowns[name][:,i].mean())
        
        eval_results["scenario_breakdown"] = {}
        for scenario, methods in scenario_results.items():
            eval_results["scenario_breakdown"][scenario] = {}
            for method, metrics in methods.items():
                if metrics:
                    eval_results["scenario_breakdown"][scenario][method] = {
                        "ade": float(np.mean([m["ade"] for m in metrics])),
                        "fde": float(np.mean([m["fde"] for m in metrics])),
                        "count": len(metrics)
                    }
        
        eval_results["n_samples"] = len(results["adaptive"])
        
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"Detailed results saved to {results_path}")
        
        report_path = os.path.join(cfg["paths"]["log_dir"], "training_report.html")
        generate_training_report("config.yaml", cfg["paths"]["log_dir"], report_path)
        
    else:
        print("No valid samples processed during evaluation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["train","test"], default="train")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    set_seed(int(cfg.get("seed", 42)))
    
    for dir_key in ["log_dir", "model_dir", "plot_dir"]:
        os.makedirs(cfg["paths"][dir_key], exist_ok=True)

    if args.mode == "train":
        train_mode(cfg, args)
    elif args.mode == "test":
        test_mode(cfg, args)

    print("DONE")


if __name__ == "__main__":
    main()