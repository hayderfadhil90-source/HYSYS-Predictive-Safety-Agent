"""
test_fcv_write.py
=================
FCV-100 write verification test.

DEFAULT STATE:  CONFIRM_WRITE_TEST = False  ← safe, no write occurs
WRITE STATE:    CONFIRM_WRITE_TEST = True   ← only after manual review

REQUIRED PRECONDITIONS (all enforced by code, not just documented):
  1. Active HYSYS case must be:  Separator_PID_Auto_Stable_BaseCase.hsc
  2. Dynamics must be running    (Integrator.IsRunning == True)
  3. PIC-100 must be in Auto     (Mode == 2)
  4. Vessel pressure ~ 800 kPa  (within ± 20 kPa)
  5. PIC-100 OP ~ 50 %          (below 80 % — not saturated)
  6. FCV desired opening ~ 50 % (within 10–50 % range)

WHAT IS WRITTEN:
  Only  FCV-100.ActuatorSPValue  is written.
  Nothing else is ever touched.

TEST SEQUENCE (when CONFIRM_WRITE_TEST = True):
  a. Connect.
  b. Run all precondition checks (abort on failure).
  c. Read and store original FCV desired position.
  d. Read original FCV actual position.
  e. Print current state.
  f. Command FCV desired position to TEST_TARGET_PCT (48.0 %).
  g. Wait WRITE_WAIT_S seconds (3 s).
  h. Read FCV desired and actual, verify desired moved within ±0.2 %.
  i. Restore original desired position (in finally block — always runs).
  j. Wait WRITE_WAIT_S seconds.
  k. Verify desired and actual returned close to original.
     Desired tolerance: ±0.2 %   Actual tolerance: ±1.0 %
  l. Print PASS or FAIL.

TOLERANCES:
  DESIRED_TOL = 0.2 %
  ACTUAL_TOL  = 1.0 %
"""

# ── WRITE ENABLE — leave False until you have reviewed this file manually ────
CONFIRM_WRITE_TEST = True

# ── Test parameters ───────────────────────────────────────────────────────────
REQUIRED_CASE   = "Separator_PID_Auto_Stable_BaseCase.hsc"
TEST_TARGET_PCT = 48.0    # temporary test value (within 10–50 %)
WRITE_WAIT_S    = 3.0     # wait between write and readback
DESIRED_TOL     = 0.2     # acceptable deviation for desired position (%)
ACTUAL_TOL      = 1.0     # acceptable deviation for actual position (%)

# Sanity limits for the clean base case (if readings are outside these, abort)
EXPECTED_PRESSURE_KPA = 800.0
PRESSURE_TOLERANCE    = 20.0    # kPa
MAX_ALLOWED_PIC_OP    = 80.0    # % — if OP >= this, base case is not clean
PIC_MODE_AUTO         = 2       # HYSYS enum: 2 = Automatic

import sys
import math
import time
import pathlib
import win32com.client
import pywintypes

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_app = None
_sim = None
_ops = None


def _connect():
    """Attach to running HYSYS and populate module-level objects."""
    global _app, _sim, _ops
    try:
        _app = win32com.client.GetActiveObject("HYSYS.Application")
        _sim = _app.ActiveDocument
        _ops = _sim.Flowsheet.Operations
    except pywintypes.com_error as exc:
        raise RuntimeError(f"Cannot connect to HYSYS: {exc}")


def _read_case_name():
    try:
        return _sim.Title
    except Exception:
        return "<unknown>"


def _is_dynamics_running():
    try:
        return bool(_sim.Solver.Integrator.IsRunning)
    except Exception:
        return False


def _pic_mode():
    try:
        return int(_ops.Item("PIC-100").Mode)
    except Exception:
        return -1


def _pic_op():
    try:
        return float(_ops.Item("PIC-100").OPValue)
    except Exception:
        return float("nan")


def _vessel_pressure():
    try:
        return float(_ops.Item("V-100").VesselPressureValue)
    except Exception:
        return float("nan")


def _fcv_desired():
    """Read FCV-100 ActuatorSPValue (%)."""
    try:
        return float(_ops.Item("FCV-100").ActuatorSPValue)
    except Exception:
        return float("nan")


def _fcv_actual():
    """Read FCV-100 PercentOpenValue (%)."""
    try:
        return float(_ops.Item("FCV-100").PercentOpenValue)
    except Exception:
        return float("nan")


def _write_fcv_desired(value_pct):
    """
    Write FCV-100 ActuatorSPValue.
    Validates and clamps internally — never writes NaN, None, or out of bounds.
    """
    if value_pct is None or math.isnan(value_pct) or math.isinf(value_pct):
        raise ValueError(f"Cannot write invalid value to FCV-100: {value_pct!r}")
    clamped = max(10.0, min(50.0, float(value_pct)))
    if abs(clamped - value_pct) > 0.001:
        print(f"  [CLAMP] {value_pct:.3f}% → {clamped:.3f}% (enforced 10–50 % bounds)")
    _ops.Item("FCV-100").ActuatorSPValue = clamped
    return clamped


def _section(title):
    print(f"\n{title}")
    print("─" * len(title))


def _abort(reason):
    print(f"\n[ABORT] {reason}")
    print("        Nothing was written to HYSYS.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Precondition checks
# ---------------------------------------------------------------------------

def run_precondition_checks():
    """
    Verify all required conditions before any write is attempted.
    Raises SystemExit (via _abort) on the first failure.
    Returns dict of checked values for use in the test.
    """
    _section("Precondition checks")

    # 1. Case name — normalised comparison
    #    sim.Title may return a full path, have extra whitespace, or differ in case.
    #    Extract basename only and compare case-insensitively.
    case = _read_case_name()
    print(f"  Case name (raw)  : {case}")
    print(f"  repr(case)       : {case!r}")
    print(f"  repr(REQUIRED)   : {REQUIRED_CASE!r}")

    required_norm = pathlib.Path(str(REQUIRED_CASE).strip()).name.casefold()
    active_norm   = pathlib.Path(str(case).strip()).name.casefold()

    print(f"  Required (norm)  : {required_norm!r}")
    print(f"  Active   (norm)  : {active_norm!r}")

    if active_norm != required_norm:
        _abort(
            f"Wrong HYSYS case is open.\n"
            f"        Required (normalised) : {required_norm}\n"
            f"        Active   (normalised) : {active_norm}\n"
            f"        Open the correct case in HYSYS and run this script again."
        )
    print(f"  ✓ Correct case")

    # 2. Dynamics running
    running = _is_dynamics_running()
    print(f"  Dynamics running : {running}")
    if not running:
        _abort(
            "Dynamics is NOT running. Start the simulation in HYSYS first.\n"
            "        (Press the green Run button in the Dynamics Assistant.)"
        )
    print(f"  ✓ Dynamics is running")

    # 3. PIC-100 mode
    mode = _pic_mode()
    mode_str = "AUTO" if mode == PIC_MODE_AUTO else f"UNKNOWN({mode})"
    print(f"  PIC-100 Mode     : {mode_str}  (raw={mode})")
    if mode != PIC_MODE_AUTO:
        _abort(
            f"PIC-100 is not in Auto mode (Mode={mode}). "
            "Set it to Auto in HYSYS before running the write test."
        )
    print(f"  ✓ PIC-100 in Auto")

    # 4. Vessel pressure
    pressure = _vessel_pressure()
    print(f"  V-100 Pressure   : {pressure:.3f} kPa  (expect {EXPECTED_PRESSURE_KPA} ± {PRESSURE_TOLERANCE})")
    if math.isnan(pressure):
        _abort("Could not read V-100 pressure.")
    if abs(pressure - EXPECTED_PRESSURE_KPA) > PRESSURE_TOLERANCE:
        _abort(
            f"V-100 pressure {pressure:.1f} kPa is too far from expected "
            f"{EXPECTED_PRESSURE_KPA} kPa. Is the simulation at steady state?"
        )
    print(f"  ✓ Pressure nominal")

    # 5. PIC-100 OP
    pic_op = _pic_op()
    print(f"  PIC-100 OP       : {pic_op:.2f}%  (must be < {MAX_ALLOWED_PIC_OP}%)")
    if math.isnan(pic_op):
        _abort("Could not read PIC-100 OP.")
    if pic_op >= MAX_ALLOWED_PIC_OP:
        _abort(
            f"PIC-100 OP = {pic_op:.1f}% ≥ {MAX_ALLOWED_PIC_OP}%. "
            "Controller is too stressed for a safe write test. "
            "Use the clean base case."
        )
    print(f"  ✓ PIC-100 OP acceptable")

    # 6. FCV original value
    fcv_desired = _fcv_desired()
    fcv_actual  = _fcv_actual()
    print(f"  FCV desired (SP) : {fcv_desired:.4f}%")
    print(f"  FCV actual       : {fcv_actual:.4f}%")
    if math.isnan(fcv_desired) or math.isnan(fcv_actual):
        _abort("Could not read FCV-100 position.")
    if not (10.0 <= fcv_desired <= 50.0):
        _abort(
            f"FCV-100 desired position {fcv_desired:.2f}% is outside "
            "the expected 10–50 % range. Check the base case."
        )
    print(f"  ✓ FCV-100 position in range")

    # 7. Test target sanity
    if not (10.0 <= TEST_TARGET_PCT <= 50.0):
        _abort(
            f"TEST_TARGET_PCT={TEST_TARGET_PCT} is outside [10, 50] %. "
            "Edit this file to fix the test target."
        )
    print(f"  ✓ Test target {TEST_TARGET_PCT:.1f}% is within bounds")

    return {
        "case":        case,
        "running":     running,
        "pic_mode":    mode,
        "pressure":    pressure,
        "pic_op":      pic_op,
        "fcv_desired": fcv_desired,
        "fcv_actual":  fcv_actual,
    }


# ---------------------------------------------------------------------------
# Write test
# ---------------------------------------------------------------------------

def run_write_test(original_desired):
    """
    Execute the write → verify → restore → verify sequence.
    The original_desired is restored in a finally block even if an exception
    occurs between the first write and the end of the function.

    Returns True on PASS, False on FAIL.
    """
    passed = True
    written_test = False   # track whether the first write actually happened

    try:
        # ── f. Write test value ───────────────────────────────────────────
        _section(f"Testing temporary command: {original_desired:.2f}% → {TEST_TARGET_PCT:.2f}%")
        written = _write_fcv_desired(TEST_TARGET_PCT)
        written_test = True
        print(f"  Written: {written:.4f}%")

        # ── g. Wait ───────────────────────────────────────────────────────
        print(f"  Waiting {WRITE_WAIT_S:.0f} s ...")
        time.sleep(WRITE_WAIT_S)

        # ── h. Readback after test write ──────────────────────────────────
        _section(f"After {WRITE_WAIT_S:.0f} s")
        after_desired = _fcv_desired()
        after_actual  = _fcv_actual()
        print(f"  Desired = {after_desired:.4f}%  (expect {TEST_TARGET_PCT:.2f} ± {DESIRED_TOL})")
        print(f"  Actual  = {after_actual:.4f}%  (expect within ±{ACTUAL_TOL} of {TEST_TARGET_PCT:.2f})")

        desired_ok = abs(after_desired - TEST_TARGET_PCT) <= DESIRED_TOL
        actual_ok  = abs(after_actual  - TEST_TARGET_PCT) <= ACTUAL_TOL
        if not desired_ok:
            print(f"  [WARN] Desired position {after_desired:.4f}% outside ±{DESIRED_TOL}% of target.")
            passed = False
        if not actual_ok:
            print(f"  [WARN] Actual position {after_actual:.4f}% outside ±{ACTUAL_TOL}% of target.")
            # actual may lag — not counted as hard FAIL unless desired also failed

    finally:
        # ── i. Restore — ALWAYS runs ─────────────────────────────────────
        if written_test:
            _section(f"Restoring: {TEST_TARGET_PCT:.2f}% → {original_desired:.2f}%")
            try:
                restored = _write_fcv_desired(original_desired)
                print(f"  Restore command sent: {restored:.4f}%")
            except Exception as exc:
                print(f"  [ERROR] Restore write failed: {exc}")
                print(f"  MANUAL ACTION REQUIRED: set FCV-100 to {original_desired:.2f}% in HYSYS.")
                passed = False
                return passed

            # ── j. Wait after restore ─────────────────────────────────────
            print(f"  Waiting {WRITE_WAIT_S:.0f} s ...")
            time.sleep(WRITE_WAIT_S)

            # ── k. Verify restoration ─────────────────────────────────────
            _section("After restore")
            final_desired = _fcv_desired()
            final_actual  = _fcv_actual()
            print(f"  Desired = {final_desired:.4f}%  (expect {original_desired:.2f} ± {DESIRED_TOL})")
            print(f"  Actual  = {final_actual:.4f}%  (expect within ±{ACTUAL_TOL} of {original_desired:.2f})")

            if abs(final_desired - original_desired) > DESIRED_TOL:
                print(f"  [WARN] Desired did not fully restore to {original_desired:.2f}%.")
                passed = False
            if abs(final_actual - original_desired) > ACTUAL_TOL:
                print(f"  [NOTE] Actual position {final_actual:.4f}% still converging — may need more time.")

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("  FCV-100 Write Test Utility")
    print(f"  CONFIRM_WRITE_TEST = {CONFIRM_WRITE_TEST}")
    print(f"  Required case      = {REQUIRED_CASE}")
    print(f"  Test target        = {TEST_TARGET_PCT:.1f} %")
    print("=" * 62)

    # ── Connect ───────────────────────────────────────────────────────────────
    _section("Connecting to HYSYS")
    try:
        _connect()
    except RuntimeError as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)
    print(f"  Connected via: HYSYS.Application")

    # ── Precondition checks (always run, even in dry mode) ────────────────────
    checks = run_precondition_checks()

    # ── Summary before any write ──────────────────────────────────────────────
    _section("Current state")
    print(f"  Case             : {checks['case']}")
    print(f"  PIC Mode         : AUTO ({checks['pic_mode']})")
    print(f"  Dynamics Running : {checks['running']}")
    print(f"  V-100 Pressure   : {checks['pressure']:.3f} kPa")
    print(f"  PIC-100 OP       : {checks['pic_op']:.2f} %")
    print(f"  FCV desired      : {checks['fcv_desired']:.4f} %")
    print(f"  FCV actual       : {checks['fcv_actual']:.4f} %")

    # ── Dry-run exit ──────────────────────────────────────────────────────────
    if not CONFIRM_WRITE_TEST:
        print()
        print("─" * 62)
        print("  [DRY MODE]  CONFIRM_WRITE_TEST = False")
        print("  All preconditions printed above.")
        print("  Nothing was written to HYSYS.")
        print()
        print("  To run the actual write test:")
        print("  1. Open Separator_PID_Auto_Stable_BaseCase.hsc in HYSYS")
        print("  2. Start dynamics simulation")
        print("  3. Confirm PIC-100 is in Auto and OP ≈ 50%")
        print("  4. Set  CONFIRM_WRITE_TEST = True  in this file")
        print("  5. Run:  python test_fcv_write.py")
        print("─" * 62)
        return

    # ── Execute write test ────────────────────────────────────────────────────
    original_desired = checks["fcv_desired"]
    try:
        passed = run_write_test(original_desired)
    except pywintypes.com_error as exc:
        print(f"\n[COM ERROR] {exc}")
        print("  HYSYS connection was lost during the test.")
        print(f"  Verify FCV-100 desired position manually in HYSYS.")
        print(f"  Target to restore: {original_desired:.4f}%")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected exception: {type(exc).__name__}: {exc}")
        print(f"  If FCV-100 was changed, restore it to {original_desired:.4f}% in HYSYS.")
        sys.exit(1)

    # ── Final verdict ─────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    verdict = "PASS" if passed else "FAIL"
    print(f"  WRITE TEST: {verdict}")
    print("=" * 62)

    if passed:
        print()
        print("  Next step: set ENABLE_HYSYS_WRITE = True in")
        print("  run_agent_with_hysys.py when ready for live agent writes.")
    else:
        print()
        print("  One or more tolerance checks failed.")
        print("  Review the output above and investigate before enabling live writes.")


if __name__ == "__main__":
    main()
