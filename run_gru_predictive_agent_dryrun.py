"""
run_gru_predictive_agent_dryrun.py
====================================
LIVE predictive-agent DRY_RUN using the plant-realistic GRU-C (no
fault_severity_pct, no elapsed_sim_time_s):

    HYSYS live signals -> 10 s rolling history -> GRU-C predicts P(t+10 s)
                       -> PredictiveAgentGRU (820 kPa, 3-sample debounce)
                       -> logs WOULD_REDUCE_FEED_20 / HOLD only

SAFETY (hard-coded):  DRY_RUN = True   ENABLE_HYSYS_WRITE = False
No FCV write function is imported anywhere. PIC-100 is never modified. VLV-100 is
touched only by the operator's manual fault at t=10 s. Needs live HYSYS with
Separator_PID_Auto_Stable_BaseCase.hsc loaded and an operator present.

HONESTY NOTES (also printed at run time and in the report):
  * The GRU-C used here is scenario C (trained on 10..65 % plugging). The official
    live test severity, 50 %, IS INSIDE that training range, so this live run
    validates plumbing/timing/live-vs-offline consistency and decision behaviour,
    NOT generalisation to an unseen severity. A live 70 % run would be the
    unseen-severity test.
  * "Actual P(t+10)" is the pressure 10 samples later (1 sample = 1 RunFor(1 s)),
    matching the training target definition; the true elapsed-time gap is logged.
  * Offline benchmark numbers are never used as live results.

Run:   python run_gru_predictive_agent_dryrun.py
"""
from __future__ import annotations
import csv, datetime, math, pathlib, sys, time
import numpy as np
import pywintypes

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hysys_interface as _hi
from hysys_interface import connect_to_hysys, stop_simulation, step_simulation, get_simulation_time_s
# set_fcv_opening_percent is DELIBERATELY NOT imported.
from predictive_agent_gru import GRUCPredictor, LiveHistoryBuffer, PredictiveAgentGRU

# ── hard-coded safety guards ─────────────────────────────────────────────────
DRY_RUN            = True
ENABLE_HYSYS_WRITE = False

# ── experiment constants ─────────────────────────────────────────────────────
REQUIRED_CASE            = "Separator_PID_Auto_Stable_BaseCase.hsc"
OUT_DIR                  = pathlib.Path(__file__).parent
TOTAL_SIM_DURATION_S     = 120.0
STEP_S                   = 1.0
FAULT_TRIGGER_ELAPSED_S  = 10.0
FAULT_RAMP_S             = 2.0
FAULT_PLUGGING_PCT       = 50            # operator instruction ONLY - never a model input
ABORT_PRESSURE_KPA       = 900.0
PRE_P_LO, PRE_P_HI       = 795.0, 805.0
PRE_OP_LO, PRE_OP_HI     = 40.0, 60.0
PRE_FCV_LO, PRE_FCV_HI   = 45.0, 55.0
RULE_BASED_TRIGGER_S     = 17.0          # reference from the documented rule-based agent run
THRESHOLDS               = (820.0,)
RELIABLE_ERR_KPA         = 2.0           # provisional "reliable" definition (Phase 10)
RELIABLE_CONSEC          = 3
ONSET_EXCLUDE_SAMPLES    = 3             # unavoidable fault-onset samples excluded from the max-error criterion
USEFUL_LEAD_S            = 2.0           # provisional: earlier than the rule-based reference by >= this
JITTER_LIMIT_KPA         = 5.0           # provisional stability limit on |dP_pred| between consecutive predictions

FIELDNAMES = [
    "timestamp", "simulation_time_s", "elapsed_sim_time_s", "pressure_now_kpa", "pressure_pred_10s_kpa",
    "actual_pressure_plus10_kpa", "forecast_error_kpa", "feed_flow_kmol_h", "gas_out_flow_kmol_h",
    "liquid_out_flow_kmol_h", "temperature_c", "pic_op_percent", "fcv_actual_percent", "vlv_actual_percent",
    "prediction_over_820", "debounce_count", "agent_action", "fault_active", "dry_run",
]
# HYSYS reads -> feature-name mapping (feature order itself comes from the saved config)
SIGNAL_KEYS = {"pressure_kpa": "pressure_now_kpa", "feed_flow_kmol_h": "feed_flow_kmol_h",
               "gas_out_flow_kmol_h": "gas_out_flow_kmol_h", "liquid_out_flow_kmol_h": "liquid_out_flow_kmol_h",
               "vessel_temperature_c": "temperature_c", "pic_op_percent": "pic_op_percent",
               "fcv_actual_percent": "fcv_actual_percent", "vlv_actual_percent": "vlv_actual_percent"}


def _f(x, default=math.nan):
    try: return float(x)
    except (TypeError, ValueError): return default

def _fmt(x, d=3):
    v = _f(x); return "" if math.isnan(v) else f"{v:.{d}f}"

def _intg_is_running():
    try: return bool(_hi._sim.Solver.Integrator.IsRunning)
    except Exception: return None


def check_preconditions() -> bool:
    ops, ok = _hi._ops, True
    print("\n[Precondition check]\n" + "-" * 60)
    def rep(flag, msg):
        nonlocal ok
        print(f"  {'[OK]' if flag else '[FAIL]'} {msg}"); ok &= bool(flag)
    try:
        case = pathlib.Path(str(_hi._sim.Title).strip()).name.casefold()
        rep(case == REQUIRED_CASE.casefold(), f"Case: {case} (need {REQUIRED_CASE.casefold()})")
    except Exception as ex: rep(False, f"case name unreadable: {ex}")
    try:
        P = float(ops.Item("V-100").VesselPressureValue); rep(PRE_P_LO <= P <= PRE_P_HI, f"Pressure {P:.2f} kPa (need {PRE_P_LO}-{PRE_P_HI})")
        mode = int(ops.Item("PIC-100").Mode); OP = float(ops.Item("PIC-100").OPValue)
        rep(mode == 2, f"PIC-100 Mode {mode} (need 2=Auto)"); rep(PRE_OP_LO <= OP <= PRE_OP_HI, f"PIC OP {OP:.2f}% (need {PRE_OP_LO}-{PRE_OP_HI})")
        fa = float(ops.Item("FCV-100").PercentOpenValue); rep(PRE_FCV_LO <= fa <= PRE_FCV_HI, f"FCV actual {fa:.2f}% (need ~50)")
        va = float(ops.Item("VLV-100").PercentOpenValue); rep(45.0 <= va <= 55.0, f"VLV-100 actual {va:.2f}% (need ~50, plugging inactive)")
    except Exception as ex: rep(False, f"read error: {ex}")
    rep(DRY_RUN and not ENABLE_HYSYS_WRITE, "Write guards: DRY_RUN=True, ENABLE_HYSYS_WRITE=False")
    print("-" * 60)
    return ok


def read_signals(ops):
    fs = _hi._sim.Flowsheet.MaterialStreams
    return {
        "pressure_now_kpa": float(ops.Item("V-100").VesselPressureValue),
        "temperature_c": float(ops.Item("V-100").VesselTemperatureValue),
        "pic_op_percent": float(ops.Item("PIC-100").OPValue),
        "pic_mode": int(ops.Item("PIC-100").Mode),
        "fcv_actual_percent": float(ops.Item("FCV-100").PercentOpenValue),
        "vlv_actual_percent": float(ops.Item("VLV-100").PercentOpenValue),
        "feed_flow_kmol_h": float(fs.Item(4).MolarFlowValue) * 3600.0,
        "gas_out_flow_kmol_h": float(fs.Item(0).MolarFlowValue) * 3600.0,
        "liquid_out_flow_kmol_h": float(fs.Item(1).MolarFlowValue) * 3600.0,
    }


def run():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / f"gru_predictive_dryrun_{ts}.csv"
    ops = _hi._ops

    predictor = GRUCPredictor()
    feats = predictor.features
    assert not any(f in ("fault_severity_pct", "elapsed_sim_time_s") for f in feats)
    agent = PredictiveAgentGRU()
    buf = LiveHistoryBuffer(feats, predictor.lookback)
    print(f"\n[NOTE] Scenario-C GRU-C was trained on 10..65% plugging. This live test at {FAULT_PLUGGING_PCT}% is "
          f"IN-SAMPLE severity: it validates the live pipeline, not unseen-severity generalisation.")
    print(f"[Phase 4/5] horizon={agent.PREDICTION_HORIZON_S:.0f}s  threshold={agent.WARNING_THRESHOLD_KPA} kPa  "
          f"debounce={agent.DEBOUNCE_SAMPLES}  DRY_RUN={DRY_RUN}  ENABLE_HYSYS_WRITE={ENABLE_HYSYS_WRITE}")

    status, aborted = "RUNNING", False
    fault_prompted = fault_active = False
    fault_onset_elapsed = fault_onset_idx = None
    trigger_idx = None
    rows: list[dict] = []
    prev_elapsed = 0.0
    csvf = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csvf, fieldnames=FIELDNAMES); writer.writeheader(); csvf.flush()

    try:
        stop_simulation()
        t0 = get_simulation_time_s()
        if t0 is None:
            status = "ABORTED_COM_ERROR"; raise RuntimeError("no t0")
        print(f"\n[Phase 6] t0={t0:.3f}s. Stepping {TOTAL_SIM_DURATION_S:.0f} x {STEP_S}s. "
              f"Predictions start once {predictor.lookback} samples are buffered.\n")
        while True:
            try: step_simulation(STEP_S)
            except pywintypes.com_error as ce:
                print(f"\n[ABORT] COM error during RunFor: {ce}"); status, aborted = "ABORTED_COM_ERROR", True; break
            sim_t = get_simulation_time_s()
            if sim_t is None:
                print("\n[ABORT] cannot read simulation time"); status, aborted = "ABORTED_COM_ERROR", True; break
            elapsed = sim_t - t0
            if elapsed - prev_elapsed > 2.0:
                print(f"\n[ABORT] sim time jumped {elapsed - prev_elapsed:.2f}s in one step (max 2.0)")
                status, aborted = "INVALID_SIM_TIME_STEP_JUMP", True; break
            prev_elapsed = elapsed
            try: s = read_signals(ops)
            except Exception as ex:
                print(f"\n[ABORT] COM read error at t={elapsed:.1f}s: {ex}"); status, aborted = "ABORTED_COM_ERROR", True; break
            if not all(math.isfinite(v) for v in s.values()):
                print(f"\n[ABORT] NaN/Inf in HYSYS reading at t={elapsed:.1f}s"); status, aborted = "ABORTED_COM_ERROR", True; break
            P = s["pressure_now_kpa"]
            if P >= ABORT_PRESSURE_KPA:
                print(f"\n[STOP] pressure {P:.2f} kPa >= {ABORT_PRESSURE_KPA:.0f} at t={elapsed:.1f}s"); status, aborted = "ABORTED_PRESSURE_LIMIT", True
            elif s["pic_mode"] != 2:
                print(f"\n[ABORT] PIC-100 left Auto (Mode={s['pic_mode']})"); status, aborted = "ABORTED_PIC_LEFT_AUTO", True

            # ── Phase 3: live history buffer (feature order from the saved config) ──
            pred = math.nan; dec = {"action": "WARMUP", "prediction_over_820": False, "debounce_count": agent.debounce_count}
            buf.push({f: s[SIGNAL_KEYS[f]] if SIGNAL_KEYS[f] in s else s[f] for f in feats})
            if not aborted and buf.full():
                try: pred = predictor.predict(buf.window())
                except Exception as ex:
                    print(f"\n[ABORT] model error at t={elapsed:.1f}s: {ex}"); status, aborted = "ABORTED_MODEL_ERROR", True
                if not aborted and not math.isfinite(pred):
                    print(f"\n[ABORT] model returned NaN/Inf at t={elapsed:.1f}s"); status, aborted = "ABORTED_MODEL_ERROR", True
            if not aborted and math.isfinite(pred):
                dec = agent.decide(pred)
                margin = pred - agent.WARNING_THRESHOLD_KPA
                print(f"t={elapsed:3.0f}s\nP_now = {P:.1f} kPa\nP_pred_10s = {pred:.1f} kPa\n"
                      f"forecast_margin = {margin:+.1f} kPa\nAction = {dec['action']} (debounce {dec['debounce_count']}/{agent.DEBOUNCE_SAMPLES})\n")
                if dec["action"] == "WOULD_REDUCE_FEED_20" and trigger_idx is None:
                    trigger_idx = len(rows)
                    print(">>> GRU-C PREDICTIVE TRIGGER (DRY_RUN - no write):\n"
                          f"    P_now={P:.1f} kPa  P_pred_10s={pred:.1f} kPa  Horizon=10 s  Action=WOULD_REDUCE_FEED_20\n")

            row = {"timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
                   "simulation_time_s": f"{sim_t:.3f}", "elapsed_sim_time_s": f"{elapsed:.3f}",
                   "pressure_now_kpa": f"{P:.3f}", "pressure_pred_10s_kpa": _fmt(pred),
                   "actual_pressure_plus10_kpa": "", "forecast_error_kpa": "",
                   "feed_flow_kmol_h": f"{s['feed_flow_kmol_h']:.3f}", "gas_out_flow_kmol_h": f"{s['gas_out_flow_kmol_h']:.3f}",
                   "liquid_out_flow_kmol_h": f"{s['liquid_out_flow_kmol_h']:.3f}", "temperature_c": f"{s['temperature_c']:.3f}",
                   "pic_op_percent": f"{s['pic_op_percent']:.3f}", "fcv_actual_percent": f"{s['fcv_actual_percent']:.3f}",
                   "vlv_actual_percent": f"{s['vlv_actual_percent']:.3f}", "prediction_over_820": dec["prediction_over_820"],
                   "debounce_count": dec["debounce_count"], "agent_action": dec["action"],
                   "fault_active": fault_active, "dry_run": DRY_RUN}
            writer.writerow(row); csvf.flush(); rows.append(row)

            # ── fault prompt (after logging the t=10 row, exactly as in the dataset protocol) ──
            if elapsed >= FAULT_TRIGGER_ELAPSED_S and not fault_prompted and not aborted:
                fault_prompted = True
                stop_simulation(); time.sleep(0.05)
                if _intg_is_running(): time.sleep(0.2); stop_simulation()
                print("[OK] Integrator stopped" if not _intg_is_running() else "[WARN] Integrator may still be running")
                pause_t = get_simulation_time_s() or sim_t
                print("\n  " + "=" * 58 + f"\n  Set VLV-100 Valve Plugging = {FAULT_PLUGGING_PCT}%\n  Ramp = {FAULT_RAMP_S:.0f} s\n"
                      "  Delay = 0\n  Active = ON\n  DO NOT press Run\n  " + "-" * 58 +
                      "\n  1. DO NOT press HYSYS Run.\n  2. Open VLV-100 -> Dynamics -> Malfunction.\n"
                      f"  3. Set Valve Plugging = {FAULT_PLUGGING_PCT}%, Ramp = {FAULT_RAMP_S:.0f} s (Delay 0).\n"
                      "  4. Activate the malfunction.\n  5. Return to PowerShell.\n  6. Press Enter.\n"
                      "  The script resumes stepping itself.\n  " + "=" * 58 + "\n")
                try: input(f"  [WAITING] Press Enter after applying {FAULT_PLUGGING_PCT}% plugging ... ")
                except EOFError:
                    print("\n[ABORT] stdin not interactive"); status, aborted = "INCOMPLETE_BEFORE_FAULT", True; break
                jump = (get_simulation_time_s() or pause_t) - pause_t
                if jump > 0.5:
                    print(f"\n[ABORT] sim time advanced {jump:.1f}s during the manual pause (<=0.5 allowed). Did you press Run?")
                    status, aborted = "INVALID_SIM_TIME_ADVANCED_DURING_MANUAL_FAULT", True; break
                fault_active, fault_onset_elapsed, fault_onset_idx = True, elapsed, len(rows) - 1
                print(f"  [OK] Fault confirmed at t={elapsed:.1f}s (sim-time advance during pause {jump:.3f}s). Resuming.\n")
            if aborted: break
            if elapsed >= TOTAL_SIM_DURATION_S: status = "COMPLETED_VALID"; break
    except KeyboardInterrupt:
        print("\n[INTERRUPT]"); status = "INCOMPLETE_AFTER_FAULT" if fault_active else "INCOMPLETE_BEFORE_FAULT"
    except Exception as ex:
        print(f"\n[ABORT] unexpected {type(ex).__name__}: {ex}"); status, aborted = (status if status != "RUNNING" else "ABORTED_COM_ERROR"), True
    finally:
        csvf.close(); print(f"\n[CSV] live rows closed: {log_path} ({len(rows)} rows)")

    analyse_and_report(rows, log_path, ts, status, fault_onset_elapsed, fault_onset_idx, trigger_idx, predictor, agent)


# ═══════════════════════════════════════════════════════════════════════════
# Analysis (Phases 8-10, 13-15)
# ═══════════════════════════════════════════════════════════════════════════

def analyse_and_report(rows, log_path, ts, status, onset_elapsed, onset_idx, trigger_idx, predictor, agent):
    H = predictor.horizon
    n = len(rows)
    P = np.array([_f(r["pressure_now_kpa"]) for r in rows]); T = np.array([_f(r["elapsed_sim_time_s"]) for r in rows])
    pred = np.array([_f(r["pressure_pred_10s_kpa"]) for r in rows])
    actual = np.full(n, np.nan); target_t = np.full(n, np.nan)
    for j in range(n):
        if j + H < n and math.isfinite(pred[j]):
            actual[j] = P[j + H]; target_t[j] = T[j + H]
            rows[j]["actual_pressure_plus10_kpa"] = f"{actual[j]:.3f}"
            rows[j]["forecast_error_kpa"] = f"{pred[j] - actual[j]:.3f}"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"[CSV] backfilled with actual P(t+10) and rewritten: {log_path}")

    err = pred - actual
    ok = np.isfinite(err)
    idx_ok = np.where(ok)[0]
    gap = (target_t[ok] - T[ok]) if ok.any() else np.array([])

    def stats(mask):
        e = err[mask & ok]
        if e.size == 0: return dict(n=0, rmse=math.nan, mae=math.nan, maxe=math.nan, bias=math.nan)
        return dict(n=int(e.size), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                    maxe=float(np.max(np.abs(e))), bias=float(np.mean(e)))
    allm = np.ones(n, bool)
    S_all = stats(allm)
    S_trans = stats((target_t >= 10.0) & (target_t <= 40.0))
    post = np.arange(n) >= (onset_idx if onset_idx is not None else n)
    S_post = stats(post)
    excl = np.zeros(n, bool)
    if onset_idx is not None: excl[onset_idx:onset_idx + ONSET_EXCLUDE_SAMPLES] = True
    S_noonset = stats(~excl)

    # threshold classification @820
    th = 820.0
    a_ev, p_ev = actual > th, pred > th
    tp = int(np.sum(ok & a_ev & p_ev)); fp = int(np.sum(ok & ~a_ev & p_ev))
    fn = int(np.sum(ok & a_ev & ~p_ev)); tn = int(np.sum(ok & ~a_ev & ~p_ev))
    prec = tp / (tp + fp) if tp + fp else math.nan; rec = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec > 0) else math.nan
    acc = (tp + tn) / max(1, tp + tn + fp + fn)

    # ── Phase 9: early warning ──────────────────────────────────────────────
    first_valid = next((T[i] for i in range(n) if math.isfinite(pred[i])), None)
    first_over = next((T[i] for i in range(n) if math.isfinite(pred[i]) and pred[i] > th), None)
    t_trig = T[trigger_idx] if trigger_idx is not None else None
    t810 = next((T[i] for i in range(n) if P[i] > 810.0), None)
    t820 = next((T[i] for i in range(n) if P[i] > 820.0), None)
    def first_confirmed(vals, thr):
        c = 0
        for i in range(n):
            c = c + 1 if vals[i] > thr else 0
            if c >= agent.DEBOUNCE_SAMPLES: return T[i]
        return None
    t_cur820_conf = first_confirmed(P, 820.0); t_cur810_conf = first_confirmed(P, 810.0)
    lead_actual820 = (t820 - t_trig) if (t_trig is not None and t820 is not None) else None
    lead_rule = (RULE_BASED_TRIGGER_S - t_trig) if t_trig is not None else None
    lead_cur820 = (t_cur820_conf - t_trig) if (t_trig is not None and t_cur820_conf is not None) else None

    print("\n" + "=" * 76 + "\n  PHASE 8 - LIVE forecast validation (predicted P(t+10) vs actual, this run only)\n" + "=" * 76)
    print(f"  validated forecasts: {S_all['n']}   (mean time gap of 10 samples = {gap.mean():.3f}s)" if gap.size else "  none validated")
    print(f"  RMSE={S_all['rmse']:.4f}  MAE={S_all['mae']:.4f}  max|err|={S_all['maxe']:.3f}  bias={S_all['bias']:+.4f} kPa")
    print(f"  transient (target t 10-40s): RMSE={S_trans['rmse']:.4f} (n={S_trans['n']})")
    print(f"  post-fault (issue >= onset): RMSE={S_post['rmse']:.4f}  MAE={S_post['mae']:.4f} (n={S_post['n']})")
    print(f"  excluding first {ONSET_EXCLUDE_SAMPLES} fault-onset samples: RMSE={S_noonset['rmse']:.4f} MAE={S_noonset['mae']:.4f} "
          f"max|err|={S_noonset['maxe']:.3f}")
    print(f"  threshold {th:.0f} kPa: accuracy={acc:.3f} precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}")

    print("\n" + "=" * 76 + "\n  PHASE 9 - Early-warning analysis\n" + "=" * 76)
    fmt = lambda v: "none" if v is None else f"{v:.1f}s"
    print(f"  first valid GRU prediction : {fmt(first_valid)}")
    print(f"  first P_pred_10s > 820     : {fmt(first_over)}")
    print(f"  debounce-confirmed trigger : {fmt(t_trig)}")
    if trigger_idx is not None:
        print(f"     P_now at trigger={P[trigger_idx]:.2f} kPa  P_pred_10s={pred[trigger_idx]:.2f} kPa  "
              f"actual P(t+10)={actual[trigger_idx]:.2f} kPa" if math.isfinite(actual[trigger_idx])
              else f"     P_now at trigger={P[trigger_idx]:.2f}  P_pred={pred[trigger_idx]:.2f}  actual not yet available")
    print(f"  actual P_now crossing 810  : {fmt(t810)}   crossing 820: {fmt(t820)}")
    print(f"  rule-based reference       : {RULE_BASED_TRIGGER_S:.1f}s")
    print(f"  current-pressure logic (P_now>820 x{agent.DEBOUNCE_SAMPLES}) confirmed at: {fmt(t_cur820_conf)}   (P_now>810 x3: {fmt(t_cur810_conf)})")
    print(f"  lead vs actual P>820 event : {'n/a' if lead_actual820 is None else f'{lead_actual820:+.1f}s'}")
    print(f"  lead vs rule-based trigger : {'n/a' if lead_rule is None else f'{lead_rule:+.1f}s'}")
    print(f"  lead vs current-pressure 820 logic: {'n/a' if lead_cur820 is None else f'{lead_cur820:+.1f}s'}")

    # ── Phase 10: fault-onset table ─────────────────────────────────────────
    print("\n" + "=" * 76 + "\n  PHASE 10 - Fault-onset analysis (issue times ~10-20 s)\n" + "=" * 76)
    latency = None
    if onset_idx is not None:
        print(f"  {'t(s)':>6}{'P_now':>10}{'P_pred_10s':>12}{'actual P(t+10)':>16}{'error':>9}")
        for i in range(onset_idx, n):
            if T[i] > (onset_elapsed + 10.5): break
            e = err[i]
            print(f"  {T[i]:6.1f}{P[i]:10.2f}{pred[i]:12.2f}{actual[i]:16.2f}{e:+9.2f}")
        for i in range(onset_idx, n - RELIABLE_CONSEC + 1):
            seg = err[i:i + RELIABLE_CONSEC]
            if np.all(np.isfinite(seg)) and np.all(np.abs(seg) <= RELIABLE_ERR_KPA):
                latency = T[i] - onset_elapsed; break
        print(f"\n  Reliability (|err| <= {RELIABLE_ERR_KPA} kPa for {RELIABLE_CONSEC} consecutive predictions): "
              + (f"reached {latency:.1f}s after fault onset" if latency is not None else "NOT reached"))
    else:
        print("  fault was never confirmed - no onset analysis")

    # ── GO / NO-GO ──────────────────────────────────────────────────────────
    post_idx = [i for i in range(n) if onset_idx is not None and i >= onset_idx + ONSET_EXCLUDE_SAMPLES + 4 and math.isfinite(pred[i])]
    jit = max((abs(pred[i] - pred[i - 1]) for i in post_idx if math.isfinite(pred[i - 1])), default=math.nan)
    valid = status == "COMPLETED_VALID"
    criteria = {
        f"live RMSE <= 1.5 kPa (all validated forecasts)": S_all["rmse"] <= 1.5,
        "live MAE <= 1.2 kPa": S_all["mae"] <= 1.2,
        "recall@820 >= 0.90": rec == rec and rec >= 0.90,
        "false negatives <= 2": fn <= 2,
        f"max error < 10 kPa excluding first {ONSET_EXCLUDE_SAMPLES} onset samples": S_noonset["maxe"] < 10.0,
        f"predictive trigger useful (>= {USEFUL_LEAD_S}s earlier than rule-based ref and before actual 820 crossing)":
            lead_rule is not None and lead_rule >= USEFUL_LEAD_S and lead_actual820 is not None and lead_actual820 > 0,
        f"no unstable predictions (max |dP_pred| between consecutive predictions <= {JITTER_LIMIT_KPA} kPa after onset)":
            jit == jit and jit <= JITTER_LIMIT_KPA,
        "run valid (completed, no abort)": valid,
    }
    go = all(criteria.values())
    print("\n" + "=" * 76 + "\n  PHASE 14 - GO / NO-GO for a future controlled FCV write test (writes NOT enabled)\n" + "=" * 76)
    for k, v in criteria.items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    verdict = "GO FOR NEXT CONTROL TEST" if go else "NO-GO"
    print(f"\n  DECISION: {verdict}")
    print("  CAUTION: 50% is inside the scenario-C training range (in-sample severity) and the model is a single seed;")
    print("  a GO here does not prove unseen-severity generalisation. Writes are never enabled by this script.")

    plots = make_plots(rows, P, T, pred, actual, target_t, err, onset_elapsed, t_trig, t810, t820, latency)

    # ── Phase 15 final report ────────────────────────────────────────────────
    print("\n" + "=" * 76 + f"\n  GRU-C PREDICTIVE AGENT DRY RUN: {'VALID' if valid else 'INVALID'}\n" + "=" * 76)
    print(f"  run status                 : {status}")
    print(f"  model                      : {predictor.model_path.name}   (regenerated={predictor.config.get('regenerated')})")
    print(f"  scalers                    : {predictor.fscaler_path.name}, {predictor.pscaler_path.name}")
    print(f"  features                   : {predictor.features}")
    print(f"  lookback / horizon         : {predictor.lookback}s / {predictor.horizon}s")
    print(f"  fault time                 : {fmt(onset_elapsed)}  ({FAULT_PLUGGING_PCT}% plugging, manual)")
    print(f"  first predictive trigger   : {fmt(t_trig)}")
    if trigger_idx is not None:
        print(f"  P_now at trigger           : {P[trigger_idx]:.3f} kPa")
        print(f"  predicted P(t+10)          : {pred[trigger_idx]:.3f} kPa")
        print(f"  actual P(t+10)             : {actual[trigger_idx]:.3f} kPa" if math.isfinite(actual[trigger_idx]) else "  actual P(t+10)             : n/a")
    print(f"  rule-based trigger ref     : {RULE_BASED_TRIGGER_S:.1f}s")
    print(f"  lead gained vs rule-based  : {'n/a' if lead_rule is None else f'{lead_rule:+.1f}s'}")
    print(f"  live RMSE / MAE            : {S_all['rmse']:.4f} / {S_all['mae']:.4f} kPa")
    print(f"  max error / bias           : {S_all['maxe']:.3f} kPa (excl. onset {S_noonset['maxe']:.3f}) / {S_all['bias']:+.4f} kPa")
    print(f"  820 precision/recall/F1    : {prec:.3f} / {rec:.3f} / {f1:.3f}")
    print(f"  false negatives / positives: {fn} / {fp}")
    print(f"  reliability latency        : {'n/a' if latency is None else f'{latency:.1f}s after fault onset'}")
    print(f"  GO / NO-GO                 : {verdict}")
    print(f"  plots                      : {[p.name for p in plots]}")
    print(f"\n  DRY_RUN={DRY_RUN}  ENABLE_HYSYS_WRITE={ENABLE_HYSYS_WRITE}  (unchanged; no writes occurred)")

    rep = OUT_DIR / f"gru_predictive_dryrun_report_{ts}.txt"
    rep.write_text("\n".join([
        "GRU-C PREDICTIVE AGENT DRY RUN - REPORT", "=" * 60, f"run_status: {status}", f"csv: {log_path.name}",
        f"model: {predictor.model_path.name} (regenerated={predictor.config.get('regenerated')})",
        f"features: {predictor.features}", f"fault_onset_s: {onset_elapsed}", f"trigger_s: {t_trig}",
        f"lead_vs_rule_based_s: {lead_rule}", f"lead_vs_actual_820_s: {lead_actual820}",
        f"live_rmse: {S_all['rmse']}", f"live_mae: {S_all['mae']}", f"max_error: {S_all['maxe']}",
        f"max_error_excl_onset: {S_noonset['maxe']}", f"bias: {S_all['bias']}",
        f"recall820: {rec}", f"precision820: {prec}", f"f1_820: {f1}", f"FN: {fn}", f"FP: {fp}",
        f"reliability_latency_s: {latency}", f"decision: {verdict}",
        "criteria: " + "; ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in criteria.items()),
        "note: in-sample severity (scenario C trained on 10..65%); single seed 42; not an unseen-severity test."]) + "\n",
        encoding="utf-8")
    print(f"[Report] {rep}")
    print("\nGRU-C PREDICTIVE AGENT DRY RUN: COMPLETE")


def make_plots(rows, P, T, pred, actual, target_t, err, onset, t_trig, t810, t820, latency):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("\n" + "=" * 76 + "\n  PHASE 13 - Plots\n" + "=" * 76)
    out = []
    def marks(ax, trig=True):
        if onset is not None: ax.axvline(onset, color="#8c564b", ls="--", lw=1.0, label=f"Fault t={onset:.0f}s")
        ax.axvline(RULE_BASED_TRIGGER_S, color="#7f7f7f", ls=":", lw=1.2, label=f"Rule-based ref t={RULE_BASED_TRIGGER_S:.0f}s")
        if trig and t_trig is not None: ax.axvline(t_trig, color="#d62728", ls="-.", lw=1.4, label=f"GRU-C trigger t={t_trig:.1f}s")
        if t820 is not None: ax.axvline(t820, color="#ff7f0e", ls="--", lw=1.0, label=f"Actual P>820 t={t820:.1f}s")
    def fin(fig, ax, name, xl, yl, title):
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, lw=0.4, alpha=0.5)
        fig.tight_layout(); p = OUT_DIR / name; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); out.append(p); print(f"  Saved: {p}")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(T, P, color="#1f77b4", lw=2.0, label="P_now"); ax.plot(T, pred, color="#2ca02c", lw=1.6, ls="--", label="P_pred_10s (issued at t)")
    ax.axhline(820, color="#9467bd", ls=":", lw=1.0, label="820 kPa"); marks(ax)
    fin(fig, ax, "gru_live_pressure_now_vs_pred10.png", "Elapsed simulation time (s)", "Pressure (kPa)", "GRU-C live: P_now vs P_pred_10s")
    fig, ax = plt.subplots(figsize=(11, 5))
    m = np.isfinite(actual)
    ax.plot(target_t[m], actual[m], color="#1f77b4", lw=2.0, label="Actual P at target time"); ax.plot(target_t[m], pred[m], color="#2ca02c", lw=1.6, ls="--", label="GRU-C forecast")
    ax.axhline(820, color="#9467bd", ls=":", lw=1.0); marks(ax, trig=False)
    fin(fig, ax, "gru_live_pred10_vs_actual10.png", "Target time (issue time + 10 s)", "Pressure (kPa)", "GRU-C live: forecast vs actual P(t+10)")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(T[m], err[m], color="#d62728", lw=1.6, label="error (pred - actual)"); ax.axhline(0, color="#555", lw=1.0)
    ax.axhspan(-RELIABLE_ERR_KPA, RELIABLE_ERR_KPA, color="#2ca02c", alpha=0.12, label=f"+/-{RELIABLE_ERR_KPA} kPa"); marks(ax)
    fin(fig, ax, "gru_live_forecast_error.png", "Issue time (s)", "Forecast error (kPa)", "GRU-C live: forecast error vs issue time")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(T, P, color="#1f77b4", lw=2.0, label="P_now"); ax.axhline(810, color="#ff7f0e", ls=":", lw=1.0, label="810 kPa"); ax.axhline(820, color="#9467bd", ls=":", lw=1.0, label="820 kPa")
    if t810 is not None: ax.axvline(t810, color="#ff7f0e", ls="--", lw=1.0, alpha=0.7, label=f"Actual P>810 t={t810:.1f}s")
    marks(ax)
    fin(fig, ax, "gru_live_trigger_timeline.png", "Elapsed simulation time (s)", "Pressure (kPa)", "GRU-C live: trigger timeline")
    fig, ax = plt.subplots(figsize=(11, 5))
    if onset is not None:
        w = (T >= onset - 1) & (T <= onset + 11)
        ax.plot(T[w], P[w], color="#1f77b4", lw=2.0, marker="o", ms=4, label="P_now")
        ax.plot(T[w], pred[w], color="#2ca02c", lw=1.6, ls="--", marker="s", ms=4, label="P_pred_10s")
        w2 = w & np.isfinite(actual); ax.plot(T[w2], actual[w2], color="#ff7f0e", lw=1.6, marker="^", ms=4, label="actual P(t+10)")
        if latency is not None: ax.axvline(onset + latency, color="#2ca02c", ls=":", lw=1.2, label=f"reliable from t={onset + latency:.0f}s")
    marks(ax); ax.set_xlim(max(0, (onset or 10) - 1), (onset or 10) + 11)
    fin(fig, ax, "gru_live_fault_onset_10_20s.png", "Issue time (s)", "Pressure (kPa)", "GRU-C live: fault-onset window (t=10-20 s)")
    return out


def main():
    print("=" * 76 + "\n  run_gru_predictive_agent_dryrun.py - GRU-C live predictive DRY_RUN\n" + "=" * 76)
    print(f"  DRY_RUN={DRY_RUN}  ENABLE_HYSYS_WRITE={ENABLE_HYSYS_WRITE}  (no FCV write is possible from this script)")
    print(f"  Case: {REQUIRED_CASE}   Official test: {FAULT_PLUGGING_PCT}% plugging at t={FAULT_TRIGGER_ELAPSED_S:.0f}s")
    try: connect_to_hysys()
    except Exception as ex: print(f"\n[ABORT] cannot connect to HYSYS: {ex}"); sys.exit(1)
    if not check_preconditions(): print("\n[ABORT] preconditions failed"); sys.exit(1)
    run()


if __name__ == "__main__":
    main()
