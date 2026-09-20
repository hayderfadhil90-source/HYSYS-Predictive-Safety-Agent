"""
predictive_agent_gru.py
=========================
Loads the plant-realistic GRU-C (scenario C, variant C: NO fault_severity_pct,
NO elapsed_sim_time_s) and provides a live history buffer plus the simple
predictive decision rule. Contains NO HYSYS code and NO write capability.

Artifacts (ONLY these; the loader refuses anything else):
  gru_extended_results/gru_C_C_seed42.pt               model weights (regenerated - see config)
  gru_extended_results/gru_C_C_feature_scaler.pkl      per-feature StandardScaler (8 features)
  gru_extended_results/gru_C_C_pressure_scaler.pkl     target scaler
  gru_extended_results/gru_C_C_config.json             feature order / lookback / horizon

Do NOT confuse with gru_C_A_* (scenario C, all-features model that uses
fault_severity_pct and elapsed_sim_time_s).

The model is a direct 10-s-ahead forecaster: input = last 10 one-second samples
[t-9 .. t] of the 8 plant signals (oldest first), output = P(t+10 s). No inputs
are extrapolated.
"""
from __future__ import annotations
import collections, json, math, pathlib, pickle, sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from train_horizon10_forecast_benchmark import SeqModel   # exact trained architecture

ARTIFACT_DIR = HERE / "gru_extended_results"
CONFIG_FILE  = "gru_C_C_config.json"
FORBIDDEN_FEATURES = ("fault_severity_pct", "elapsed_sim_time_s")


class GRUCPredictor:
    def __init__(self, artifact_dir: pathlib.Path | str = ARTIFACT_DIR, verbose: bool = True):
        d = pathlib.Path(artifact_dir)
        cfg_path = d / CONFIG_FILE
        if not cfg_path.exists():
            raise FileNotFoundError(f"GRU-C config not found: {cfg_path}")
        self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
        c = self.config
        if not (c.get("scenario") == "C" and c.get("variant") == "GRU-C"):
            raise ValueError("Config is not scenario C / variant GRU-C")

        self.features = list(c["features"])                       # order comes from the saved config
        bad = [f for f in self.features if f in FORBIDDEN_FEATURES or "above" in f or "error" in f]
        if bad:
            raise ValueError(f"Forbidden/non-plant features in GRU-C config: {bad}")
        self.lookback = int(c["lookback_s"]); self.horizon = int(c["horizon_s"])

        self.model_path = d / c["model_file"]
        self.fscaler_path = d / c["feature_scaler_file"]
        self.pscaler_path = d / c["pressure_scaler_file"]
        for p in (self.model_path, self.fscaler_path, self.pscaler_path):
            if not p.exists():
                raise FileNotFoundError(f"Required GRU-C artifact missing: {p}")
        for p, must in ((self.model_path, "gru_C_C_"), (self.fscaler_path, "gru_C_C_"), (self.pscaler_path, "gru_C_C_")):
            if not p.name.startswith(must):
                raise ValueError(f"Wrong artifact (expected scenario C / variant C): {p.name}")

        with open(self.fscaler_path, "rb") as f: self.fscaler = pickle.load(f)
        with open(self.pscaler_path, "rb") as f: self.pscaler = pickle.load(f)
        if self.fscaler.n_features_in_ != len(self.features):
            raise ValueError(f"Feature scaler expects {self.fscaler.n_features_in_} features but config lists "
                             f"{len(self.features)} - wrong scaler file?")

        self.model = SeqModel(len(self.features), hidden=int(c["gru_hidden"]))
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model.eval()

        if verbose:
            print("[GRUCPredictor] loaded (scenario C / variant C = plant-realistic GRU-C):")
            print(f"   model          : {self.model_path}")
            print(f"   feature scaler : {self.fscaler_path}")
            print(f"   target scaler  : {self.pscaler_path}")
            print(f"   config/schema  : {cfg_path}")
            print(f"   features ({len(self.features)}): {self.features}")
            print(f"   lookback={self.lookback}s  horizon={self.horizon}s  hidden={c['gru_hidden']}  "
                  f"seed={c['seed']}  regenerated={c.get('regenerated')}")

    def predict(self, window: np.ndarray) -> float:
        """window: (lookback, n_features), oldest first, physical units, columns in self.features order."""
        w = np.asarray(window, dtype=np.float64)
        if w.shape != (self.lookback, len(self.features)):
            raise ValueError(f"window shape {w.shape} != {(self.lookback, len(self.features))}")
        if not np.all(np.isfinite(w)):
            raise ValueError("non-finite values in input window")
        wn = self.fscaler.transform(w).reshape(1, self.lookback, len(self.features))
        with torch.no_grad():
            pn = self.model(torch.tensor(wn, dtype=torch.float32)).numpy()
        return float(self.pscaler.inverse_transform(pn).ravel()[0])


class LiveHistoryBuffer:
    """Rolling buffer of the latest `lookback` valid samples (1 sim-s each)."""
    def __init__(self, features: list, lookback: int):
        self.features, self.lookback = features, lookback
        self._buf = collections.deque(maxlen=lookback)

    def push(self, sample: dict):
        vec = [float(sample[f]) for f in self.features]
        if not all(math.isfinite(v) for v in vec):
            raise ValueError("non-finite sample rejected")
        self._buf.append(vec)

    def full(self) -> bool:
        return len(self._buf) == self.lookback

    def window(self) -> np.ndarray:
        return np.array(self._buf, dtype=np.float64)


class PredictiveAgentGRU:
    """Simple DRY_RUN decision rule. Never writes anything."""
    PREDICTION_HORIZON_S = 10.0
    WARNING_THRESHOLD_KPA = 820.0
    DEBOUNCE_SAMPLES = 3

    def __init__(self):
        self.debounce_count = 0

    def decide(self, p_pred_10s: float) -> dict:
        over = p_pred_10s > self.WARNING_THRESHOLD_KPA
        self.debounce_count = self.debounce_count + 1 if over else 0
        action = "WOULD_REDUCE_FEED_20" if self.debounce_count >= self.DEBOUNCE_SAMPLES else "HOLD"
        return {"action": action, "prediction_over_820": over, "debounce_count": self.debounce_count}


if __name__ == "__main__":
    p = GRUCPredictor()
    import pandas as pd
    df = pd.read_csv(HERE / "hysys_pinn_dataset.csv")
    g = df[df.fault_severity_pct == 70].sort_values("elapsed_sim_time_s").reset_index(drop=True)
    i = 60
    w = g.loc[i - 9:i, p.features].to_numpy()
    print(f"offline sanity (70%, window ending row {i}): pred={p.predict(w):.3f}  actual P(t+10)={g.loc[i+10,'pressure_kpa']:.3f}")
