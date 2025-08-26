import math
import numpy as np
from typing import Dict, List, Iterator
import torch
from torch.utils.data import Sampler
from trajdata.data_structures.batch import AgentBatch

def _angle_deg(vec):
    return math.degrees(math.atan2(float(vec[1]), float(vec[0]) + 1e-8))

def classify_sample(hist_xy: np.ndarray,
                    fut_xy: np.ndarray,
                    angle_deg_thresh: float = 12.0,
                    speed_limit_ms: float = 16.7) -> Dict[str, str | bool]:
    p0 = hist_xy[0]; p1 = hist_xy[-1]; pf = fut_xy[-1]
    v_hist = p1 - p0
    v_fut  = pf - p1
    a0 = _angle_deg(v_hist); a1 = _angle_deg(v_fut)
    d = a1 - a0
    while d > 180: d -= 360
    while d < -180: d += 360
    if d > angle_deg_thresh:
        man = "turn_left"
    elif d < -angle_deg_thresh:
        man = "turn_right"
    else:
        man = "straight"

    T = max(1, fut_xy.shape[0])
    dist = float(np.linalg.norm(fut_xy[-1] - fut_xy[0]))
    vavg = dist / (0.5 * T)
    overspeed = vavg > float(speed_limit_ms)

    return {"maneuver": man, "overspeed": overspeed, "stopline": False}


def _positions(t):
    if hasattr(t, "positions"):
        arr = t.positions
    elif hasattr(t, "position"):
        arr = t.position
    else:
        arr = t[..., :2]
    return arr.detach().cpu().numpy().astype(np.float32)

def estimate_distribution(dataset,
                          angle_deg_thresh: float = 12.0,
                          speed_limit_ms: float = 16.7,
                          num_samples: int = 2000,
                          seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    N = len(dataset)
    counts = {"straight": 0, "turn_left": 0, "turn_right": 0, "stopline": 0, "overspeed": 0}
    collate = dataset.get_collate_fn(pad_format="right")

    used = 0
    for _ in range(min(num_samples, N)):
        idx = int(rng.integers(0, N))
        try:
            batch: AgentBatch = collate([dataset[idx]])
        except Exception:
            continue
        if batch is None or getattr(batch, "agent_hist", None) is None:
            continue
        try:
            hist = _positions(batch.agent_hist[0])
            fut  = _positions(batch.agent_fut[0])
        except Exception:
            continue

        cls = classify_sample(hist, fut, angle_deg_thresh, speed_limit_ms)
        counts[cls["maneuver"]] += 1
        if cls.get("overspeed", False): counts["overspeed"] += 1
        if cls.get("stopline",  False): counts["stopline"]  += 1
        used += 1

    if used == 0:
        return {"straight": 0.6, "turn_left": 0.2, "turn_right": 0.2, "stopline": 0.0, "overspeed": 0.0}

    main = ["straight", "turn_left", "turn_right", "stopline"]
    total_main = sum(counts[k] for k in main)
    if total_main == 0:
        total_main = 1
    dist = {k: counts[k] / float(total_main) for k in main}
    dist["overspeed"] = counts["overspeed"] / max(1, used)
    return dist


class ProportionalCategorySampler(Sampler[int]):
    def __init__(self,
                 dataset,
                 proportions: Dict[str, float],
                 angle_deg_thresh: float = 12.0,
                 speed_limit_ms: float = 16.7,
                 seed: int = 42,
                 scan_cap: int = 4000):
        self.dataset = dataset
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.proportions = self._norm_main(proportions)
        self.angle_deg_thresh = float(angle_deg_thresh)
        self.speed_limit_ms = float(speed_limit_ms)

        self.buckets: Dict[str, List[int]] = {"straight": [], "turn_left": [], "turn_right": [], "stopline": []}
        self._build_buckets(scan_cap)

    def _norm_main(self, p: Dict[str, float]) -> Dict[str, float]:
        main = {k: p.get(k, 0.0) for k in ["straight", "turn_left", "turn_right", "stopline"]}
        s = sum(main.values())
        if s <= 0:
            return {"straight": 1.0, "turn_left": 0.0, "turn_right": 0.0, "stopline": 0.0}
        return {k: v / s for k, v in main.items()}

    def _build_buckets(self, cap: int):
        collate = self.dataset.get_collate_fn(pad_format="right")
        N = len(self.dataset)
        K = min(cap, N)
        idxs = self.rng.permutation(N)[:K]
        for idx in idxs:
            try:
                b = collate([self.dataset[int(idx)]])
            except Exception:
                continue
            if b is None or getattr(b, "agent_hist", None) is None:
                continue
            try:
                h = _positions(b.agent_hist[0]); f = _positions(b.agent_fut[0])
            except Exception:
                continue
            tag = classify_sample(h, f, self.angle_deg_thresh, self.speed_limit_ms)["maneuver"]
            if tag not in self.buckets: tag = "straight"
            self.buckets[tag].append(int(idx))
        
        universe = list(range(N))
        for k, v in self.buckets.items():
            if len(v) == 0:
                self.buckets[k] = [int(x) for x in self.rng.choice(universe, size=min(64, N), replace=False)]

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        cats = ["straight", "turn_left", "turn_right", "stopline"]
        probs = np.array([self.proportions[c] for c in cats], dtype=np.float64)
        probs /= probs.sum()
        for _ in range(len(self)):
            c = str(self.rng.choice(cats, p=probs))
            bucket = self.buckets.get(c, None)
            if not bucket:
                yield int(self.rng.integers(0, len(self.dataset)))
            else:
                yield int(bucket[self.rng.integers(0, len(bucket))])