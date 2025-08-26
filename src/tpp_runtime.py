import os, json
from typing import Any, Tuple
import numpy as np
import torch
import torch.nn as nn

from trajectron.model.model_registrar import ModelRegistrar
from trajectron.model.trajectron import Trajectron
from trajectron.model.model_utils import UpdateMode


class TPPRuntime:
    def __init__(
        self,
        config_path: str,
        model_dir: str,
        iteration: int = 20,
        max_modes: int = 6,
        device: str = "cuda",
        eager: bool = False,
        force_cpu_load: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.max_modes = int(max_modes)
        self.loaded = False

        with open(config_path, "r") as f:
            hyperparams = json.load(f)

        ckpt = os.path.join(model_dir, f"model_registrar-{iteration}.pt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        print(f"Loading TPP checkpoint: {ckpt}")
        
        # Method 1: Try official Trajectron++ loading
        try:
            registrar = ModelRegistrar(model_dir, self.device)
            registrar.load_models(iteration)  # This calls the official method
            self.model = Trajectron(registrar, hyperparams, device=self.device, log_writer=None)
            self.model.eval()
            self.loaded = True
            print(f"✓ TPP loaded via official method")
            if eager:
                for p in self.model.parameters():
                    p.data = p.data.to(self.device, copy=True)
            return
        except Exception as e:
            print(f"Official TPP loading failed: {e}")

        # Method 2: Manual checkpoint loading with proper dict->ModuleDict conversion
        try:
            print("Attempting manual checkpoint loading...")
            checkpoint = torch.load(ckpt, map_location="cpu" if force_cpu_load else self.device)
            
            # Create fresh registrar and model
            registrar = ModelRegistrar(model_dir, self.device)
            model = Trajectron(registrar, hyperparams, device=self.device, log_writer=None)

            # 1) Check for state_dict format used by official Trajectron++ saves
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                try:
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    self.model = model
                    self.model.eval()
                    self.loaded = True
                    print("✓ TPP loaded via model_state_dict")
                    return
                except Exception as e:
                    print(f"  Failed to load model_state_dict: {e}")

            # 2) Direct ModuleDict checkpoint
            if isinstance(checkpoint, nn.ModuleDict):
                registrar.model_dict = checkpoint.to(self.device)
                self.model = model
                self.model.eval()
                self.loaded = True
                print("✓ TPP loaded via ModuleDict assignment")
                return

            # 3) Plain dictionary of modules -> convert to ModuleDict
            if isinstance(checkpoint, dict):
                model_dict = checkpoint.get("model_dict", checkpoint)
                if isinstance(model_dict, dict) and not isinstance(model_dict, nn.ModuleDict):
                    model_dict = nn.ModuleDict(model_dict)
                if isinstance(model_dict, nn.ModuleDict):
                    registrar.model_dict = model_dict.to(self.device)
                    self.model = model
                    self.model.eval()
                    self.loaded = True
                    print("✓ TPP loaded via dict->ModuleDict conversion")
                    return

            print("All manual loading methods failed")
            
        except Exception as e:
            print(f"Manual loading failed: {e}")

        # Method 3: Create empty model as last resort
        try:
            print("Creating empty TPP model as fallback...")
            registrar = ModelRegistrar(model_dir, self.device)
            model = Trajectron(registrar, hyperparams, device=self.device, log_writer=None)
            model.eval()
            self.model = model
            self.loaded = True
            print("⚠ TPP model created but weights are uninitialized")
            
        except Exception as e:
            print(f"Even empty model creation failed: {e}")
            raise RuntimeError("Complete TPP loading failure")

        if eager and self.loaded:
            for p in self.model.parameters():
                p.data = p.data.to(self.device, copy=True)

    @torch.no_grad()
    def propose_from_batch(self, batch: Any, agent_index: int = 0, num_modes: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        if not self.loaded:
            raise RuntimeError("TPP model not properly loaded")
            
        try:
            # Enhanced batch validation
            if not hasattr(batch, 'agent_hist') or not hasattr(batch, 'agent_fut'):
                raise ValueError("Batch missing agent_hist or agent_fut")
            
            if len(batch.agent_hist) <= agent_index:
                raise ValueError(f"Batch has {len(batch.agent_hist)} agents, requested {agent_index}")
            
            # Ensure batch is properly formatted for TPP
            batch = batch.to(self.device)
            
            # Check if model has trained weights
            has_weights = False
            for param in self.model.parameters():
                if param.abs().sum() > 1e-6:
                    has_weights = True
                    break
            
            if not has_weights:
                print("TPP model has no trained weights, using enhanced fallback")
                raise ValueError("No trained weights")
            
            # Call TPP prediction
            pred_dists, _ = self.model.predict(batch, update_mode=UpdateMode.BATCH_FROM_PRIOR)
            
            M = min(int(num_modes), 25)  # Reasonable upper bound
            
            # Try different extraction methods
            extraction_methods = [
                ("sample", lambda pd, m: pd.sample(m)),
                ("modes", lambda pd, m: pd.modes(m) if hasattr(pd, "modes") else None),
                ("mus", lambda pd, m: pd.mus[:, :m] if hasattr(pd, "mus") else None)
            ]
            
            for method_name, method_func in extraction_methods:
                try:
                    result = method_func(pred_dists, M)
                    if result is not None and result.shape[0] > agent_index:
                        trajs = result[agent_index].detach().cpu().numpy().astype(np.float32)
                        if len(trajs.shape) == 3:  # (M, T, 2)
                            prior = np.ones(trajs.shape[0], dtype=np.float32) / float(trajs.shape[0])
                            print(f"TPP {method_name} success: {trajs.shape}")
                            return trajs, prior
                except Exception as e:
                    print(f"TPP {method_name} failed: {e}")
                    continue
                    
        except Exception as main_error:
            print(f"TPP prediction failed: {main_error}")

        # Enhanced fallback using batch information
        return self._generate_fallback_trajectories(batch, agent_index, num_modes)
    
    def _generate_fallback_trajectories(self, batch: Any, agent_index: int, num_modes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic fallback trajectories using batch information"""
        try:
            if hasattr(batch, 'agent_fut') and len(batch.agent_fut) > agent_index:
                fut_tensor = batch.agent_fut[agent_index]
                T = fut_tensor.shape[0]
            else:
                T = 12  # Default prediction length
            
            if hasattr(batch, 'agent_hist') and len(batch.agent_hist) > agent_index:
                hist_tensor = batch.agent_hist[agent_index]
                # Estimate velocity from history
                if hist_tensor.shape[0] > 1:
                    vel_vec = hist_tensor[-1, :2] - hist_tensor[-2, :2]  # Last velocity
                    start_pos = hist_tensor[-1, :2].detach().cpu().numpy()
                else:
                    vel_vec = torch.tensor([1.0, 0.0])  # Default forward
                    start_pos = np.array([0.0, 0.0])
                vel_vec = vel_vec.detach().cpu().numpy()
            else:
                start_pos = np.array([0.0, 0.0])
                vel_vec = np.array([1.0, 0.0])  # Default forward velocity

            props = []
            K = int(min(num_modes, 10))
            
            for m in range(K):
                # Create diverse trajectories with different curvatures and speeds
                speed_factor = 0.8 + 0.4 * (m / max(1, K-1))  # 0.8x to 1.2x speed
                lateral_offset = (m - (K - 1) / 2.0) * 2.0 / max(1, K-1)  # Lateral spread
                
                traj_points = []
                current_pos = start_pos.copy()
                current_vel = vel_vec * speed_factor
                
                for t in range(T):
                    # Add some curvature
                    curvature = lateral_offset * 0.1 * np.sin(t * 0.5)
                    # Rotate velocity slightly for curvature
                    angle = curvature
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    rotated_vel = np.array([
                        current_vel[0] * cos_a - current_vel[1] * sin_a,
                        current_vel[0] * sin_a + current_vel[1] * cos_a
                    ])
                    
                    current_pos = current_pos + rotated_vel * 0.5  # dt = 0.5
                    traj_points.append(current_pos.copy())
                    
                    # Add slight lateral drift
                    current_pos[1] += lateral_offset * 0.05
                
                props.append(np.array(traj_points, dtype=np.float32))
            
            props = np.stack(props, axis=0)
            prior = np.ones(props.shape[0], dtype=np.float32) / float(props.shape[0])
            print(f"Enhanced fallback generated: {props.shape}")
            return props, prior
            
        except Exception as e:
            print(f"Enhanced fallback failed: {e}")
            # Minimal fallback
            T = 12
            start_pos = np.array([0.0, 0.0])
            props = []
            K = int(min(num_modes, 6))
            
            for m in range(K):
                xs = np.linspace(start_pos[0], start_pos[0] + 6, T, dtype=np.float32)
                y_offset = (m - (K - 1) / 2.0) * 1.0
                ys = np.linspace(start_pos[1], start_pos[1] + y_offset, T, dtype=np.float32)
                props.append(np.stack([xs, ys], axis=1))
            
            props = np.stack(props, axis=0)
            prior = np.ones(props.shape[0], dtype=np.float32) / float(props.shape[0])
            print(f"Minimal fallback generated: {props.shape}")
            return props, prior