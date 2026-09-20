"""
train_action_aware_gru.py
===========================
OFFLINE training/validation of an ACTION-AWARE GRU that forecasts separator pressure
10 s ahead both before and after a deliberate FCV-100 action.

No HYSYS connection, no live writes, no agent changes.

Data      : hysys_action_aware_combined.csv  (9 NO_ACTION + 15 CONTROLLED_ACTION runs)
Inputs    : 8 plant-observable channels x 10 s lookback (oldest -> newest), NO severity,
            NO elapsed time, NO action time / seconds-since-action / pre-post label /
            scenario label (those exist in the file for ANALYSIS only).
Target    : pressure_kpa at t+10 s (row i+10 of the same run; windows never cross runs).
Model     : GRU(hidden=32, 1 layer) -> Linear -> scalar (same size as the previous GRU-C).
Training  : Adam lr 3e-3, MSE on standardised target, full batch, early stopping
            (patience 150, max 3000), scalers fit on the TRAINING block only, seeds 42-46.
Validation: blocked late-time segment (issue time > 96 s) of TRAINING runs only.

Evaluation scenarios (test runs never appear in training or validation):
  A  controlled interpolation : test = plug50_fcv40
  B  unseen severity          : train excludes EVERY severity-70 run (controlled + no-action);
                                test = plug70_fcv35/40/45 (+ the no-action 70 run reported separately)
  C  unseen FCV target        : test = all FCV=40 controlled runs; train = FCV 35/45 + all no-action
  D  hard combined holdout    : test = plug70_fcv40 only

Regions (s = issue_time - action_time, controlled runs only; the action is at ~15.5 s and a window needs
10 s of history, so the FIRST forecast of every controlled run is issued only ~5 s before the write, and EVERY
controlled-run forecast has its target AFTER the action):
  pre         s < 0          issued before the write: inputs cannot show the action, target is affected (unavoidable blind spot)
  transition  s < 5          the blind pre-write samples plus the first 5 s while the valve moves
  post        s >= 0         SPEC DEFINITION of post-action (comparable with the old 10.692 live figure)
    bins      0-5, 5-15, >15 s
  post_visible s > 0         first window that can see the FCV movement
  no_action_runs             held-out NO_ACTION trajectories (clean pre-action-style forecasting; only exists in scenario B)
NOTE: the deterministic simulator makes the first seconds of a held-out controlled run identical to those of its
no-action / sibling runs that stay in training, so pre-action-style numbers there are optimistic; POST metrics are
the honest ones for A, C, D. B (severity 70 fully held out) is clean for everything.

Ablation (Phase 11): GRU-A all 8 features vs GRU-B without fcv_actual_percent.
Also: the OLD GRU-C (gru_C_C_seed42) is replayed offline on the same test runs as a reference.

Run:   python -u train_action_aware_gru.py           (long: run in background)
Smoke: python -u train_action_aware_gru.py --smoke-test
"""
from __future__ import annotations
import argparse, json, math, pathlib, pickle, random, sys, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
DATA = HERE / "hysys_action_aware_combined.csv"
OUT = HERE / "action_aware_gru_results"

FEATURES = ["pressure_kpa", "feed_flow_kmol_h", "gas_out_flow_kmol_h", "liquid_out_flow_kmol_h",
            "vessel_temperature_c", "pic_op_percent", "fcv_actual_percent", "vlv_actual_percent"]
FORBIDDEN = {"fault_severity_pct", "elapsed_sim_time_s", "action_time_s", "seconds_since_action",
             "pre_or_post_action", "scenario_type", "fcv_target_percent", "fcv_initial_percent", "action_executed"}
assert not (set(FEATURES) & FORBIDDEN)
VARIANTS = {"GRU-A": FEATURES, "GRU-B": [f for f in FEATURES if f != "fcv_actual_percent"]}

LOOKBACK, HORIZON = 10, 10
SEEDS = [42, 43, 44, 45, 46]
HIDDEN, LR, MAX_EPOCHS, PATIENCE = 32, 3e-3, 3000, 150   # lr 3e-3 (was 1e-3): same accuracy in ~1/2 the compute (checked on scenario D)
VAL_ISSUE_TIME_S = 96.0
THRESHOLDS = [810.0, 820.0, 830.0]
OLD_LIVE_POST_RMSE = 10.692

SCENARIOS = {
    "A": {"name": "controlled interpolation", "test": lambda m: m["run_id"] == "plug50_fcv40",
          "excluded_from_train": lambda m: m["run_id"] == "plug50_fcv40"},
    "B": {"name": "unseen severity 70", "test": lambda m: m["severity"] == 70,
          "excluded_from_train": lambda m: m["severity"] == 70},
    "C": {"name": "unseen FCV target 40", "test": lambda m: (m["scenario"] == "CONTROLLED_ACTION") & (m["fcv_target"] == 40),
          "excluded_from_train": lambda m: (m["scenario"] == "CONTROLLED_ACTION") & (m["fcv_target"] == 40)},
    "D": {"name": "hard combined holdout plug70_fcv40", "test": lambda m: m["run_id"] == "plug70_fcv40",
          "excluded_from_train": lambda m: m["run_id"] == "plug70_fcv40"},
}
REGIONS = ["all_controlled", "pre", "transition", "post", "post_0_5", "post_5_15", "post_gt15", "post_visible"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 - audit
# ═══════════════════════════════════════════════════════════════════════════

def audit() -> pd.DataFrame:
    print("\n" + "=" * 76 + "\n  PHASE 1 - Data audit\n" + "=" * 76)
    df = pd.read_csv(DATA)
    print(f"  rows: {len(df)}   trajectories: {df.run_id.nunique()}")
    dup = int(df.duplicated(["run_id", "elapsed_sim_time_s"]).sum())
    print(f"  {'[OK]' if dup == 0 else '[FAIL]'} duplicate (run, time) rows: {dup}")
    bad = {c: int((~np.isfinite(df[c].to_numpy(float))).sum()) for c in FEATURES}
    bad = {k: v for k, v in bad.items() if v}
    print(f"  {'[OK]' if not bad else '[FAIL]'} NaN/Inf in required features: {bad or 'none'}")
    mono = all(np.all(np.diff(g.sort_values("elapsed_sim_time_s").elapsed_sim_time_s.to_numpy()) > 0) for _, g in df.groupby("run_id"))
    print(f"  {'[OK]' if mono else '[FAIL]'} time strictly monotonic within every trajectory")
    print("  scenario counts (rows / runs):",
          {k: (int(v), int(df[df.scenario_type == k].run_id.nunique())) for k, v in df.scenario_type.value_counts().items()})
    print("  severity coverage (all):", sorted(int(x) for x in df.fault_severity_pct.unique()))
    ctl = df[df.scenario_type == "CONTROLLED_ACTION"]
    print("  controlled severity coverage:", sorted(int(x) for x in ctl.fault_severity_pct.unique()),
          "  FCV target coverage:", sorted(int(x) for x in ctl.fcv_target_percent.dropna().unique()))
    print("  pre-action rows:", int((ctl.pre_or_post_action == "pre_action").sum()),
          "  post-action rows:", int((ctl.pre_or_post_action == "post_action").sum()),
          "  no-action rows:", int((df.scenario_type == "NO_ACTION").sum()))
    print("  FCV actual range  no-action:", (df[df.scenario_type == 'NO_ACTION'].fcv_actual_percent.min(), df[df.scenario_type == 'NO_ACTION'].fcv_actual_percent.max()),
          " controlled:", (round(ctl.fcv_actual_percent.min(), 2), round(ctl.fcv_actual_percent.max(), 2)))
    if dup or bad or not mono:
        raise SystemExit("[ABORT] audit failed")
    print("=" * 76)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 - sequences
# ═══════════════════════════════════════════════════════════════════════════

def build_sequences(df: pd.DataFrame):
    Xs, ys, meta = [], [], []
    for rid, g in df.groupby("run_id"):
        g = g.sort_values("elapsed_sim_time_s").reset_index(drop=True)
        n = len(g)
        arr = g[FEATURES].to_numpy(float)
        sc = g.scenario_type.iloc[0]
        sev = int(g.fault_severity_pct.iloc[0])
        tgt = g.fcv_target_percent.iloc[0]
        ta = g.action_time_s.iloc[0] if sc == "CONTROLLED_ACTION" else np.nan
        for i in range(LOOKBACK - 1, n - HORIZON):
            Xs.append(arr[i - LOOKBACK + 1: i + 1]); ys.append(g.pressure_kpa.iloc[i + HORIZON])
            it = float(g.elapsed_sim_time_s.iloc[i])
            meta.append({"run_id": rid, "scenario": sc, "severity": sev, "fcv_target": float(tgt) if sc == "CONTROLLED_ACTION" else np.nan,
                         "action_time": float(ta) if sc == "CONTROLLED_ACTION" else np.nan, "issue_time": it,
                         "target_time": float(g.elapsed_sim_time_s.iloc[i + HORIZON]),
                         "s": (it - float(ta)) if sc == "CONTROLLED_ACTION" else np.nan,
                         "p_now": float(g.pressure_kpa.iloc[i])})
    return np.stack(Xs), np.array(ys), pd.DataFrame(meta)


# ═══════════════════════════════════════════════════════════════════════════
# Model / training
# ═══════════════════════════════════════════════════════════════════════════

class ActionGRU(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def fit_scalers(Xtr, ytr):
    fs = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))
    ps = StandardScaler().fit(ytr.reshape(-1, 1))
    return fs, ps

def tx(fs, X): return fs.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)


def train_model(Xtr, ytr, Xva, yva, seed):
    set_seed(seed)
    model = ActionGRU(Xtr.shape[-1]); opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    Xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32).view(-1, 1)
    best, state, wait, ep = math.inf, None, 0, 0
    for ep in range(MAX_EPOCHS):
        model.train(); opt.zero_grad()
        loss = torch.mean((model(Xt) - yt) ** 2); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = torch.mean((model(Xv) - yv) ** 2).item()
        if vl < best - 1e-9: best, state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    model.load_state_dict(state)
    return model, ep + 1, best


def predict(model, ps, Xn):
    model.eval()
    with torch.no_grad(): out = model(torch.tensor(Xn, dtype=torch.float32)).numpy()
    return ps.inverse_transform(out).ravel()


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def reg_metrics(y, p):
    if len(y) == 0: return dict(n=0, rmse=math.nan, mae=math.nan, r2=math.nan, maxerr=math.nan, bias=math.nan)
    e = p - y
    ss = float(np.sum((y - y.mean()) ** 2))
    return dict(n=int(len(y)), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                r2=float(1 - np.sum(e ** 2) / ss) if ss > 1e-12 else math.nan,
                maxerr=float(np.max(np.abs(e))), bias=float(np.mean(e)))

def region_masks(meta: pd.DataFrame):
    ctl = (meta.scenario == "CONTROLLED_ACTION").to_numpy(); s = meta.s.to_numpy(float)
    with np.errstate(invalid="ignore"):
        return {"all_controlled": ctl, "pre": ctl & (s < 0), "transition": ctl & (s < 5),
                "post": ctl & (s >= 0), "post_0_5": ctl & (s >= 0) & (s < 5), "post_5_15": ctl & (s >= 5) & (s < 15),
                "post_gt15": ctl & (s >= 15), "post_visible": ctl & (s > 0)}

def thr_metrics(y, p, th):
    a, b = y > th, p > th
    tp, fp, fn, tn = int(np.sum(a & b)), int(np.sum(~a & b)), int(np.sum(a & ~b)), int(np.sum(~a & ~b))
    prec = tp / (tp + fp) if tp + fp else math.nan; rec = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec > 0) else math.nan
    return dict(events=int(a.sum()), tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


# ═══════════════════════════════════════════════════════════════════════════
# Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario(X, y, meta, scn, variant, feat_idx, old_model=None):
    cfg = SCENARIOS[scn]
    test_m = np.asarray(cfg["test"](meta)); excl = np.asarray(cfg["excluded_from_train"](meta))
    tr_all = ~excl
    val_m = tr_all & (meta.issue_time.to_numpy() > VAL_ISSUE_TIME_S); tr_m = tr_all & ~val_m
    assert not (test_m & tr_all).any(), "test leaks into train/val"
    Xtr, ytr, Xva, yva = X[tr_m][..., feat_idx], y[tr_m], X[val_m][..., feat_idx], y[val_m]
    Xte, yte, mte = X[test_m][..., feat_idx], y[test_m], meta[test_m].reset_index(drop=True)
    fs, ps = fit_scalers(Xtr, ytr)
    Xtr_n, Xva_n, Xte_n = tx(fs, Xtr), tx(fs, Xva), tx(fs, Xte)
    ytr_n, yva_n = ps.transform(ytr.reshape(-1, 1)).ravel(), ps.transform(yva.reshape(-1, 1)).ravel()
    print(f"\n  === Scenario {scn} ({cfg['name']}) | {variant} | train seq={len(ytr)} val={len(yva)} test={len(yte)} "
          f"(test runs: {sorted(mte.run_id.unique())}) ===", flush=True)
    masks = region_masks(mte); out = {"meta": mte, "y": yte, "preds": [], "seed_metrics": [], "fs": fs, "ps": ps, "model42": None, "epochs": []}
    for seed in SEEDS:
        t0 = time.time()
        model, ep, vbest = train_model(Xtr_n, ytr_n, Xva_n, yva_n, seed)
        p = predict(model, ps, Xte_n); out["preds"].append(p); out["epochs"].append(ep)
        if seed == SEEDS[0]: out["model42"] = model
        post = reg_metrics(yte[masks["post"]], p[masks["post"]]); pre = reg_metrics(yte[masks["pre"]], p[masks["pre"]])
        print(f"    seed={seed} epochs={ep:4d} ({time.time() - t0:4.0f}s)  pre RMSE={pre['rmse']:.3f}  post RMSE={post['rmse']:.3f} "
              f"MAE={post['mae']:.3f} max={post['maxerr']:.2f} bias={post['bias']:+.3f}", flush=True)
    return out


def old_gru_predictions(X, meta, test_mask):
    from predictive_agent_gru import GRUCPredictor
    pr = GRUCPredictor(verbose=False)
    assert pr.features == FEATURES
    Xte = X[test_mask]
    Xn = pr.fscaler.transform(Xte.reshape(-1, Xte.shape[-1])).reshape(Xte.shape)
    with torch.no_grad(): out = pr.model(torch.tensor(Xn, dtype=torch.float32)).numpy()
    return pr.pscaler.inverse_transform(out).ravel()


def main():
    global SEEDS, MAX_EPOCHS, PATIENCE
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke-test", action="store_true"); ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()
    if args.smoke_test: SEEDS, MAX_EPOCHS, PATIENCE = [42, 43], 40, 10; print("[SMOKE TEST]")
    OUT.mkdir(exist_ok=True)
    df = audit()
    X, y, meta = build_sequences(df)
    print(f"\n  PHASE 2/3: features ({len(FEATURES)}): {FEATURES}\n  lookback={LOOKBACK}s horizon={HORIZON}s  sequences={len(y)}  "
          f"(controlled {int((meta.scenario == 'CONTROLLED_ACTION').sum())}, no-action {int((meta.scenario == 'NO_ACTION').sum())})")
    print("  forbidden inputs excluded:", sorted(FORBIDDEN))

    results = {}
    variants = ["GRU-A"] if args.no_ablation else ["GRU-A", "GRU-B"]
    for scn in SCENARIOS:
        for v in variants:
            idx = [FEATURES.index(f) for f in VARIANTS[v]]
            results[(scn, v)] = run_scenario(X, y, meta, scn, v, idx)

    # old GRU-C reference (offline replay on the same test sequences)
    old = {}
    for scn in SCENARIOS:
        tm = np.asarray(SCENARIOS[scn]["test"](meta)); old[scn] = old_gru_predictions(X, meta, tm)

    # ── metric tables ────────────────────────────────────────────────────────
    per_seed, agg_rows, thr_rows, bin_rows = [], [], [], []
    for (scn, v), r in results.items():
        masks = region_masks(r["meta"])
        for si, seed in enumerate(SEEDS):
            p = r["preds"][si]
            for reg in REGIONS:
                m = masks[reg]; d = reg_metrics(r["y"][m], p[m]); d.update(scenario=scn, variant=v, seed=seed, region=reg); per_seed.append(d)
            noact = (r["meta"].scenario == "NO_ACTION").to_numpy()
            if noact.any():
                d = reg_metrics(r["y"][noact], p[noact]); d.update(scenario=scn, variant=v, seed=seed, region="no_action_runs"); per_seed.append(d)
            for reg, m in (("pre_s<0", masks["all_controlled"] & (r["meta"].s.to_numpy(float) < 0)), ("post_s>=0", masks["post"])):
                for th in THRESHOLDS:
                    d = thr_metrics(r["y"][m], p[m], th); d.update(scenario=scn, variant=v, seed=seed, region=reg, threshold=th); thr_rows.append(d)
    ps_df = pd.DataFrame(per_seed); th_df = pd.DataFrame(thr_rows)
    metric_keys = ["rmse", "mae", "r2", "maxerr", "bias"]
    for (scn, v, reg), g in ps_df.groupby(["scenario", "variant", "region"]):
        d = {"scenario": scn, "variant": v, "region": reg, "n": int(g.n.iloc[0])}
        for k in metric_keys: d[k + "_mean"], d[k + "_std"] = float(g[k].mean()), float(g[k].std(ddof=0))
        d["worst_seed_rmse"], d["worst_seed_maxerr"] = float(g.rmse.max()), float(g.maxerr.max()); agg_rows.append(d)
    agg = pd.DataFrame(agg_rows)
    th_agg = th_df.groupby(["scenario", "variant", "region", "threshold"]).agg(
        events=("events", "mean"), precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
        fp=("fp", "mean"), fn=("fn", "mean"), fn_worst_seed=("fn", "max")).reset_index()

    old_rows = []
    for scn in SCENARIOS:
        r = results[(scn, "GRU-A")]; masks = region_masks(r["meta"])
        for reg in REGIONS:
            m = masks[reg]; d = reg_metrics(r["y"][m], old[scn][m]); d.update(scenario=scn, region=reg); old_rows.append(d)
    old_df = pd.DataFrame(old_rows)

    ps_df.to_csv(OUT / "action_gru_per_seed_metrics.csv", index=False)
    agg.to_csv(OUT / "action_gru_summary_metrics.csv", index=False)
    th_agg.to_csv(OUT / "action_gru_threshold_metrics.csv", index=False)
    old_df.to_csv(OUT / "action_gru_old_gruC_offline_replay_metrics.csv", index=False)

    def A(scn, v, reg): return agg[(agg.scenario == scn) & (agg.variant == v) & (agg.region == reg)].iloc[0]
    print("\n" + "=" * 128 + "\n  PHASE 6/7 - Metrics, GRU-A (all 8 features): mean +/- std over seeds; regions defined in the module docstring\n" + "=" * 128)
    print(f"  {'scn':<4}{'region':<15}{'n':>5}{'RMSE':>16}{'MAE':>16}{'R2':>9}{'maxErr':>9}{'worstSeedMax':>13}{'bias':>17}   old GRU-C RMSE")
    for scn in SCENARIOS:
        for reg in ["pre", "transition", "post", "post_0_5", "post_5_15", "post_gt15", "post_visible"]:
            a = A(scn, "GRU-A", reg); o = old_df[(old_df.scenario == scn) & (old_df.region == reg)].iloc[0]
            print(f"  {scn:<4}{reg:<15}{a.n:>5.0f}{a.rmse_mean:>9.3f}+/-{a.rmse_std:<5.3f}{a.mae_mean:>9.3f}+/-{a.mae_std:<5.3f}{a.r2_mean:>9.3f}"
                  f"{a.maxerr_mean:>9.2f}{a.worst_seed_maxerr:>13.2f}{a.bias_mean:>9.3f}+/-{a.bias_std:<5.3f}   {o.rmse:.3f}")
    if (agg.region == "no_action_runs").any():
        a = agg[(agg.region == "no_action_runs") & (agg.variant == "GRU-A")]
        for _, r in a.iterrows(): print(f"  {r.scenario:<4}{'no-action run(s)':<15}{r.n:>5.0f}{r.rmse_mean:>9.3f}+/-{r.rmse_std:<5.3f}  (held-out no-action trajectory)")

    print("\n" + "=" * 112 + "\n  PHASE 9 - Threshold decisions, GRU-A (seed-mean). 'events' = actual P(t+10) > threshold in that region\n" + "=" * 112)
    print(f"  {'scn':<4}{'region':<11}{'thr':>5}{'events':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'meanFP':>8}{'meanFN':>8}{'FN worst seed':>15}")
    for _, r in th_agg[th_agg.variant == "GRU-A"].iterrows():
        print(f"  {r.scenario:<4}{r.region:<11}{r.threshold:>5.0f}{r.events:>8.1f}{r.precision:>8.3f}{r.recall:>8.3f}{r.f1:>8.3f}{r.fp:>8.2f}{r.fn:>8.2f}{r.fn_worst_seed:>15.0f}")

    # ── Phase 8 vs old ───────────────────────────────────────────────────────
    print("\n" + "=" * 100 + f"\n  PHASE 8 - Post-action (s>=0) RMSE vs old GRU-C (live controlled write: {OLD_LIVE_POST_RMSE} kPa)\n" + "=" * 100)
    cmp_rows = []
    for scn in SCENARIOS:
        a = A(scn, "GRU-A", "post"); o = old_df[(old_df.scenario == scn) & (old_df.region == "post")].iloc[0]
        red_live, red_off = OLD_LIVE_POST_RMSE - a.rmse_mean, o.rmse - a.rmse_mean
        cmp_rows.append({"scenario": scn, "new_post_rmse": a.rmse_mean, "old_live_post_rmse": OLD_LIVE_POST_RMSE, "old_offline_replay_post_rmse": o.rmse,
                         "abs_reduction_vs_live": red_live, "pct_reduction_vs_live": red_live / OLD_LIVE_POST_RMSE * 100,
                         "abs_reduction_vs_offline_replay": red_off, "pct_reduction_vs_offline_replay": red_off / o.rmse * 100 if o.rmse else math.nan})
        print(f"  {scn}: new {a.rmse_mean:.3f}  | old live {OLD_LIVE_POST_RMSE:.3f} -> {red_live:+.3f} kPa ({red_live / OLD_LIVE_POST_RMSE * 100:+.1f}%)"
              f"  | old GRU-C offline replay on same runs {o.rmse:.3f} -> {red_off:+.3f} kPa ({red_off / o.rmse * 100 if o.rmse else float('nan'):+.1f}%)")
    pd.DataFrame(cmp_rows).to_csv(OUT / "action_gru_vs_old_gruC.csv", index=False)

    # ── Phase 10: by FCV target (scenario B: same severity, three targets) and by severity (scenario C: same FCV, five severities)
    print("\n" + "=" * 90 + "\n  PHASE 10 - Post-action error by FCV target (scenario B, 70% plugging) and by severity (scenario C, FCV=40)\n" + "=" * 90)
    by_t, by_s = [], []
    rB = results[("B", "GRU-A")]; mB = rB["meta"]; postB = region_masks(mB)["post"]
    for t in (45, 40, 35):
        m = postB & (mB.fcv_target.to_numpy() == t)
        vals = [reg_metrics(rB["y"][m], p[m]) for p in rB["preds"]]
        d = {"fcv_target": t, "n": vals[0]["n"], "rmse_mean": np.mean([v["rmse"] for v in vals]), "rmse_std": np.std([v["rmse"] for v in vals]),
             "mae_mean": np.mean([v["mae"] for v in vals]), "maxerr_mean": np.mean([v["maxerr"] for v in vals]), "bias_mean": np.mean([v["bias"] for v in vals])}
        by_t.append(d); print(f"  FCV {t}%: post RMSE={d['rmse_mean']:.3f}+/-{d['rmse_std']:.3f}  MAE={d['mae_mean']:.3f}  max={d['maxerr_mean']:.2f}  bias={d['bias_mean']:+.3f}")
    rC = results[("C", "GRU-A")]; mC = rC["meta"]; postC = region_masks(mC)["post"]
    for sv in sorted(mC.severity.unique()):
        m = postC & (mC.severity.to_numpy() == sv)
        vals = [reg_metrics(rC["y"][m], p[m]) for p in rC["preds"]]
        d = {"severity": int(sv), "n": vals[0]["n"], "rmse_mean": np.mean([v["rmse"] for v in vals]), "rmse_std": np.std([v["rmse"] for v in vals]),
             "mae_mean": np.mean([v["mae"] for v in vals]), "maxerr_mean": np.mean([v["maxerr"] for v in vals]), "bias_mean": np.mean([v["bias"] for v in vals])}
        by_s.append(d); print(f"  severity {int(sv)}% (FCV 40): post RMSE={d['rmse_mean']:.3f}+/-{d['rmse_std']:.3f}  MAE={d['mae_mean']:.3f}  max={d['maxerr_mean']:.2f}  bias={d['bias_mean']:+.3f}")
    pd.DataFrame(by_t).to_csv(OUT / "action_gru_post_error_by_fcv_target.csv", index=False)
    pd.DataFrame(by_s).to_csv(OUT / "action_gru_post_error_by_severity.csv", index=False)

    # ── Phase 11 ablation ────────────────────────────────────────────────────
    if not args.no_ablation:
        print("\n" + "=" * 100 + "\n  PHASE 11 - Ablation: GRU-A (8 features) vs GRU-B (no fcv_actual_percent), post-action (s>=0)\n" + "=" * 100)
        for scn in SCENARIOS:
            a, b = A(scn, "GRU-A", "post"), A(scn, "GRU-B", "post")
            print(f"  {scn}: A RMSE={a.rmse_mean:.3f}+/-{a.rmse_std:.3f} MAE={a.mae_mean:.3f} max={a.maxerr_mean:.2f} | "
                  f"B RMSE={b.rmse_mean:.3f}+/-{b.rmse_std:.3f} MAE={b.mae_mean:.3f} max={b.maxerr_mean:.2f}  (B/A x{b.rmse_mean / a.rmse_mean:.2f})")

    # ── Phase 14 readiness (worst case over all four scenarios, GRU-A) ──────
    print("\n" + "=" * 100 + "\n  PHASE 14 - Second-stage DRY_RUN readiness (GRU-A, post-action s>=0, must hold in ALL scenarios A-D)\n" + "=" * 100)
    verdict_all, table = True, []
    for scn in SCENARIOS:
        a = A(scn, "GRU-A", "post")
        t820 = th_agg[(th_agg.scenario == scn) & (th_agg.variant == "GRU-A") & (th_agg.region == "post_s>=0") & (th_agg.threshold == 820.0)].iloc[0]
        crit = {
            "post RMSE <= 2.0": a.rmse_mean <= 2.0, "post MAE <= 1.2": a.mae_mean <= 1.2,
            "max post error < 10 (mean and worst seed)": a.maxerr_mean < 10 and a.worst_seed_maxerr < 10,
            "recall@820 post >= 0.90": (t820.recall == t820.recall) and t820.recall >= 0.90,
            "false negatives@820 <= 2": t820.fn <= 2.0,
            "no severe seed instability (RMSE std<1, worst<2x mean)": a.rmse_std < 1.0 and a.worst_seed_rmse < 2 * a.rmse_mean,
            "no strong systematic bias (|bias|<=0.5)": abs(a.bias_mean) <= 0.5,
        }
        ok = all(crit.values()); verdict_all &= ok
        print(f"  Scenario {scn} ({SCENARIOS[scn]['name']}): {'PASS' if ok else 'FAIL'}")
        for k, v in crit.items(): print(f"      [{'PASS' if v else 'FAIL'}] {k}")
        table.append((scn, ok, crit))
    ready = verdict_all
    print(f"\n  DECISION: {'READY FOR SECOND-STAGE DRY_RUN' if ready else 'NOT READY FOR SECOND-STAGE DRY_RUN'}  "
          "(offline only; live second-stage write is NOT enabled by this script)")

    make_plots(results, old, ps_df)

    # ── artifacts: scenario models + final all-data candidate ────────────────
    for (scn, v), r in results.items():
        tag = f"action_gru_scn{scn}_{v.split('-')[1]}_seed{SEEDS[0]}"
        torch.save(r["model42"].state_dict(), OUT / f"{tag}.pt")
        with open(OUT / f"{tag}_feature_scaler.pkl", "wb") as f: pickle.dump(r["fs"], f)
        with open(OUT / f"{tag}_target_scaler.pkl", "wb") as f: pickle.dump(r["ps"], f)
    idx = list(range(len(FEATURES)))
    val_m = meta.issue_time.to_numpy() > VAL_ISSUE_TIME_S
    fs, ps = fit_scalers(X[~val_m], y[~val_m])
    print("\n  Training FINAL all-data candidate (GRU-A, seed 42; no holdout - validated only via scenarios A-D of the same recipe) ...", flush=True)
    fm, ep, _ = train_model(tx(fs, X[~val_m]), ps.transform(y[~val_m].reshape(-1, 1)).ravel(), tx(fs, X[val_m]), ps.transform(y[val_m].reshape(-1, 1)).ravel(), SEEDS[0])
    torch.save(fm.state_dict(), OUT / "action_gru_FINAL_alldata_GRU-A_seed42.pt")
    with open(OUT / "action_gru_FINAL_alldata_feature_scaler.pkl", "wb") as f: pickle.dump(fs, f)
    with open(OUT / "action_gru_FINAL_alldata_target_scaler.pkl", "wb") as f: pickle.dump(ps, f)
    config = {"features": FEATURES, "features_GRU_B": VARIANTS["GRU-B"], "lookback_s": LOOKBACK, "horizon_s": HORIZON, "gru_hidden": HIDDEN,
              "seeds": SEEDS, "lr": LR, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "val_issue_time_s": VAL_ISSUE_TIME_S,
              "scenarios": {k: v["name"] for k, v in SCENARIOS.items()}, "thresholds_kpa": THRESHOLDS, "regions": REGIONS,
              "excluded_inputs": sorted(FORBIDDEN), "final_model_file": "action_gru_FINAL_alldata_GRU-A_seed42.pt",
              "final_model_epochs": ep, "final_model_note": "trained on ALL runs (no test holdout); not independently validated",
              "old_live_post_rmse": OLD_LIVE_POST_RMSE, "smoke_test": bool(args.smoke_test)}
    (OUT / "action_gru_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ── final report ─────────────────────────────────────────────────────────
    D = A("D", "GRU-A", "post"); Dpre = A("D", "GRU-A", "pre"); Bp = A("B", "GRU-A", "post"); Bpre = A("B", "GRU-A", "pre")
    t820D = th_agg[(th_agg.scenario == "D") & (th_agg.variant == "GRU-A") & (th_agg.region == "post_s>=0") & (th_agg.threshold == 820.0)].iloc[0]
    hardest = max(SCENARIOS, key=lambda s: A(s, "GRU-A", "post").rmse_mean)
    Hh = A(hardest, "GRU-A", "post")
    lines = ["ACTION-AWARE GRU BENCHMARK - REPORT", "=" * 70, json.dumps(config, indent=2), "",
             f"Decision: {'READY' if ready else 'NOT READY'} FOR SECOND-STAGE DRY_RUN"]
    for scn, ok, crit in table:
        lines.append(f"Scenario {scn}: {'PASS' if ok else 'FAIL'} " + "; ".join(f"[{'P' if v else 'F'}] {k}" for k, v in crit.items()))
    (OUT / "action_gru_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 76 + "\n  ACTION-AWARE GRU BENCHMARK: COMPLETE\n" + "=" * 76)
    print(f"  dataset                    : {len(df)} rows, {df.run_id.nunique()} trajectories ({len(y)} sequences)")
    print(f"  architecture               : GRU-{HIDDEN} x1 -> Linear, 8 observable inputs, lookback {LOOKBACK}s, horizon {HORIZON}s")
    print(f"  scenarios                  : " + "; ".join(f"{k}={v['name']}" for k, v in SCENARIOS.items()))
    print(f"  hardest holdout (by post RMSE): scenario {hardest}: post RMSE={Hh.rmse_mean:.3f}+/-{Hh.rmse_std:.3f}, MAE={Hh.mae_mean:.3f}, max={Hh.maxerr_mean:.2f} (worst seed {Hh.worst_seed_maxerr:.2f})")
    na = agg[(agg.scenario == "B") & (agg.variant == "GRU-A") & (agg.region == "no_action_runs")]
    na_txt = f"{na.rmse_mean.iloc[0]:.3f}" if len(na) else "n/a"
    print(f"  pre-action RMSE            : clean no-action 70% (unseen severity, scenario B) = {na_txt} kPa ; pre-write blind samples D = {Dpre.rmse_mean:.3f}, B = {Bpre.rmse_mean:.3f} kPa")
    print(f"  D (plug70_fcv40)           : post RMSE={D.rmse_mean:.3f}+/-{D.rmse_std:.3f}  MAE={D.mae_mean:.3f}  max={D.maxerr_mean:.2f}  bias={D.bias_mean:+.3f}")
    print(f"  B (all 70% controlled)     : post RMSE={Bp.rmse_mean:.3f}+/-{Bp.rmse_std:.3f}  MAE={Bp.mae_mean:.3f}  max={Bp.maxerr_mean:.2f}")
    print(f"  D post recall/F1 @820      : {t820D.recall:.3f} / {t820D.f1:.3f}   false negatives: {t820D.fn:.2f} (worst seed {t820D.fn_worst_seed:.0f})   FP {t820D.fp:.2f}")
    print(f"  seed stability (D post RMSE std) : {D.rmse_std:.3f} kPa (worst seed {D.worst_seed_rmse:.3f})")
    print(f"  improvement vs old live {OLD_LIVE_POST_RMSE} kPa : D {OLD_LIVE_POST_RMSE - D.rmse_mean:+.3f} kPa ({(OLD_LIVE_POST_RMSE - D.rmse_mean) / OLD_LIVE_POST_RMSE * 100:+.1f}%)")
    print(f"  DECISION                   : {'READY FOR SECOND-STAGE DRY_RUN' if ready else 'NOT READY FOR SECOND-STAGE DRY_RUN'}")
    print(f"  artifacts                  : {OUT}")
    print("  No HYSYS connection made. No live writes enabled.")
    print("=" * 76)


def make_plots(results, old, ps_df):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("\n" + "=" * 72 + "\n  PHASE 12 - Plots\n" + "=" * 72)
    def save(fig, name):
        p = OUT / name; fig.tight_layout(); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  Saved: {p}")
    def band(scn, run_id, fname, title):
        r = results[(scn, "GRU-A")]; m = (r["meta"].run_id == run_id).to_numpy(); mt = r["meta"][m]
        preds = np.stack([p[m] for p in r["preds"]]); t = mt.target_time.to_numpy(); ta = float(mt.action_time.iloc[0])
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(t, r["y"][m], color="#1f77b4", lw=2.2, label="Actual P(t+10)")
        ax.plot(t, preds.mean(0), color="#d62728", lw=1.6, ls="--", label="Action-aware GRU (5-seed mean)")
        ax.fill_between(t, preds.min(0), preds.max(0), color="#d62728", alpha=0.18, label="seed min-max")
        ax.axvline(ta + HORIZON, color="k", ls="-.", lw=1.0, label=f"target time of action-time forecast (t={ta + HORIZON:.1f}s)")
        ax.axvline(ta, color="#2ca02c", ls=":", lw=1.2, label=f"FCV action t={ta:.1f}s"); ax.axhline(820, color="#9467bd", ls=":", lw=0.9, label="820 kPa")
        ax.set_xlabel("Target time (s)"); ax.set_ylabel("Pressure (kPa)"); ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, lw=0.4, alpha=0.5)
        save(fig, fname)
    band("A", "plug50_fcv40", "action_gru_actual_vs_pred_plug50_fcv40.png", "Held-out plug50_fcv40 (scenario A): actual vs predicted P(t+10)")
    band("D", "plug70_fcv40", "action_gru_actual_vs_pred_plug70_fcv40.png", "Held-out plug70_fcv40 (scenario D): actual vs predicted P(t+10)")
    fig, ax = plt.subplots(figsize=(11, 5)); r = results[("B", "GRU-A")]; mt = r["meta"]
    cols = {45: "#1f77b4", 40: "#2ca02c", 35: "#d62728"}
    for t_ in (45, 40, 35):
        m = ((mt.fcv_target == t_) & (mt.s >= -6)).to_numpy(); s = mt.s.to_numpy()[m]
        e = np.mean([p[m] - r["y"][m] for p in r["preds"]], axis=0); ax.plot(s, e, color=cols[t_], lw=1.8, label=f"FCV {t_}% (70% plugging, 5-seed mean)")
    eo = old["B"];
    ax.axhline(0, color="#555", lw=1.0); ax.axvline(0, color="k", ls="-.", lw=1.0, label="FCV action"); ax.axvspan(-6, 0, color="#ff7f0e", alpha=0.08, label="blind window (issued before the write)")
    ax.set_xlabel("Issue time relative to action (s)"); ax.set_ylabel("Forecast error, pred - actual (kPa)"); ax.set_title("Post-action forecast error (scenario B, unseen severity 70%)")
    ax.legend(fontsize=8); ax.grid(True, lw=0.4, alpha=0.5); save(fig, "action_gru_post_action_error.png")
    fig, ax = plt.subplots(figsize=(9, 5)); mB = r["meta"]; postB = region_masks(mB)["post"]
    for i, t_ in enumerate((45, 40, 35)):
        m = postB & (mB.fcv_target.to_numpy() == t_); vals = [reg_metrics(r["y"][m], p[m])["rmse"] for p in r["preds"]]
        ax.bar(i, np.mean(vals), yerr=np.std(vals), color=cols[t_], alpha=0.85, capsize=4); ax.scatter(np.full(len(vals), i), vals, color="k", s=12, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(["FCV 45%", "FCV 40%", "FCV 35%"]); ax.axhline(2.0, color="#2ca02c", ls=":", label="2.0 kPa criterion")
    ax.set_ylabel("Post-action RMSE (kPa)"); ax.set_title("Error by FCV target (scenario B, 70% plugging)"); ax.legend(fontsize=8); ax.grid(True, axis="y", lw=0.4, alpha=0.5)
    save(fig, "action_gru_error_by_fcv_target.png")
    fig, ax = plt.subplots(figsize=(9, 5)); rC = results[("C", "GRU-A")]; mC = rC["meta"]; postC = region_masks(mC)["post"]; sv = sorted(mC.severity.unique())
    for i, s_ in enumerate(sv):
        m = postC & (mC.severity.to_numpy() == s_); vals = [reg_metrics(rC["y"][m], p[m])["rmse"] for p in rC["preds"]]
        ax.bar(i, np.mean(vals), yerr=np.std(vals), color="#1f77b4", alpha=0.85, capsize=4); ax.scatter(np.full(len(vals), i), vals, color="k", s=12, zorder=3)
    ax.set_xticks(range(len(sv))); ax.set_xticklabels([f"{int(s)}%" for s in sv]); ax.axhline(2.0, color="#2ca02c", ls=":", label="2.0 kPa criterion")
    ax.set_xlabel("Plugging severity (FCV=40 held out entirely)"); ax.set_ylabel("Post-action RMSE (kPa)"); ax.set_title("Error by severity (scenario C, unseen FCV target 40%)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", lw=0.4, alpha=0.5); save(fig, "action_gru_error_by_severity.png")
    fig, ax = plt.subplots(figsize=(10, 5)); rng = np.random.default_rng(0)
    for i, scn in enumerate(SCENARIOS):
        g = ps_df[(ps_df.scenario == scn) & (ps_df.variant == "GRU-A") & (ps_df.region == "post")]
        ax.scatter(np.full(len(g), i) + rng.uniform(-0.08, 0.08, len(g)), g.rmse, color="#d62728", s=40, alpha=0.85, label="new GRU-A (per seed)" if i == 0 else None, zorder=3)
        ax.scatter([i], [g.rmse.mean()], color="k", marker="_", s=300, zorder=4)
        rr = results[(scn, "GRU-A")]; m = region_masks(rr["meta"])["post"]; ov = reg_metrics(rr["y"][m], old[scn][m])["rmse"]
        ax.scatter([i], [ov], color="#7f7f7f", marker="D", s=50, label="old GRU-C (offline replay)" if i == 0 else None, zorder=3)
    ax.axhline(OLD_LIVE_POST_RMSE, color="#7f7f7f", ls="--", lw=1.0, label=f"old GRU-C live {OLD_LIVE_POST_RMSE} kPa"); ax.axhline(2.0, color="#2ca02c", ls=":", label="2.0 kPa criterion")
    ax.set_xticks(range(len(SCENARIOS))); ax.set_xticklabels([f"{k}: {v['name']}" for k, v in SCENARIOS.items()], fontsize=7)
    ax.set_yscale("log"); ax.set_ylabel("Post-action RMSE (kPa, log)"); ax.set_title("Seed stability of post-action RMSE"); ax.legend(fontsize=8); ax.grid(True, lw=0.4, alpha=0.5)
    save(fig, "action_gru_seed_stability.png")


if __name__ == "__main__":
    main()
