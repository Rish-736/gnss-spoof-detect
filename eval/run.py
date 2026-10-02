"""Phase 0 entry point.

    python -m eval.run --scenario jump          # the Phase 0 deliverable
    python -m eval.run --scenario all --trials 20
    python -m eval.run --roc
    python -m eval.run --drift-sweep

Every number printed comes from repeated runs over independent seeds, not
from one lucky trial.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from detector import SpoofDetector
from sim import scenario as scenario_mod

from . import metrics, plots

RESULTS = Path(__file__).resolve().parent.parent / "results"


# --------------------------------------------------------------------------
# the streaming loop: this is the only place sim and detector touch
# --------------------------------------------------------------------------

def run_detector(scn, **det_kw) -> dict:
    """Replay one scenario through the detector, in real arrival order.

    The detector receives body-frame IMU at 100 Hz and a GPS position at
    1 Hz, and nothing else. Swapping this loop for a serial reader is the
    whole of the Phase 3 port.
    """
    det = SpoofDetector(fs_imu=scn.fs_imu, dt_gps=scn.dt_gps,
                        gps_sigma_m=scn.gps_spec.sigma_m, **det_kw)

    t_gps, flags, score, z, resid, sigma, dr_pos = [], [], [], [], [], [], []
    dt_imu = 1.0 / scn.fs_imu

    for kind, _t, *payload in scn.iter_stream():
        if kind == "imu":
            det.step_imu(payload[0], payload[1], payload[2], dt_imu)
            continue
        out = det.step_gps(payload[0], scn.dt_gps)
        t_gps.append(out["t"])
        flags.append(out["spoofed"])
        score.append(out["score"])
        z.append(out["z"])
        resid.append(out["residual_m"])
        sigma.append(out["sigma_m"])
        dr_pos.append(out["dr_pos"])

    return {
        "t_gps": np.asarray(t_gps),
        "flags": np.asarray(flags, bool),
        "score": np.asarray(score),
        "z": np.asarray(z),
        "residual_m": np.asarray(resid),
        "sigma_m": np.asarray(sigma),
        "dr_pos": np.asarray(dr_pos),
        "flag_time": det.flag_time,
        "warmup_fixes": det.warmup_fixes,
        "threshold": det.threshold,
        "cusum_k": det.cusum_k,
        "mode": det.mode,
        "detector": det,
    }


def trial(spoof, seed, path="survey", duration=240.0, det_kw=None,
          spoof_params=None):
    """One scenario + one detector run + its score."""
    preset = dict(scenario_mod.PRESETS[spoof])
    if spoof_params:
        preset["spoof_params"] = {**preset.get("spoof_params", {}),
                                  **spoof_params}
    scn = scenario_mod.build(path=path, duration=duration, seed=seed, **preset)
    log = run_detector(scn, **(det_kw or {}))
    score = metrics.score_run(log["t_gps"], log["flags"], scn.onset_s,
                              log["warmup_fixes"], kind=spoof,
                              truth_error=scn.attack.truth_error,
                              dt_gps=scn.dt_gps)
    return scn, log, score


def run_many(spoof, trials, seed0=0, **kw):
    out = []
    for i in range(trials):
        _scn, _log, s = trial(spoof, seed0 + i, **kw)
        out.append(s)
    return out


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_scenarios(args) -> None:
    kinds = (["clean", "jump", "drift", "replay"] if args.scenario == "all"
             else [args.scenario])
    det_kw = dict(mode=args.mode)

    print("\n{} trials per scenario, {}s {} flight, mode={}".format(
        args.trials, args.duration, args.path, args.mode))
    print("-" * 68)
    print(metrics.HEADER)

    summaries = []
    for kind in kinds:
        scores = run_many(kind, args.trials, args.seed, path=args.path,
                          duration=args.duration, det_kw=det_kw)
        s = metrics.summarise(scores)
        summaries.append(s)
        print(s.row())
    print("-" * 68)
    clean = next((s for s in summaries if s.kind == "clean"), summaries[0])
    print("latency: seconds from spoof onset to the first flag.\n"
          "FA/hour: false-alarm ONSETS per hour of clean flight, measured over "
          "{:.0f} min of it.".format(clean.clean_minutes))

    # One worked example per scenario, with plots.
    for kind in kinds:
        scn, log, score = trial(kind, args.seed, path=args.path,
                                duration=args.duration, det_kw=det_kw)
        name = "scenario_{}.png".format(kind)
        verdict = ("flagged at t={:.1f}s".format(log["flag_time"])
                   if log["flag_time"] is not None else "no alarm")
        plots.plot_scenario(
            scn, log, RESULTS / name,
            title="{} spoof  --  {}  (seed {})".format(kind, verdict, args.seed))
        print("  {:<7} seed {}: {:<22} peak induced error {:.1f} m  -> {}".format(
            kind, args.seed, verdict, score.peak_truth_error_m, name))

    if len(summaries) > 1:
        plots.plot_latency(summaries, RESULTS / "latency.png")
        print("  latency chart -> latency.png")


def cmd_roc(args) -> None:
    """Sweep the alarm threshold, re-running the detector at each setting.

    The threshold changes the detector's own trust gating, so the scores
    cannot be reused across thresholds -- each point is a full re-run. That
    is slower and correct.
    """
    grids = {
        "cusum": ("cusum_h", [2, 3, 4, 6, 8, 12, 16, 24, 34, 48]),
        "nsigma": ("n_sigma", [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 11.0, 15.0]),
    }
    modes = ["cusum", "nsigma"] if args.mode == "both" else [args.mode]
    attacks = [a for a in args.roc_attacks.split(",") if a]

    curves = {}
    t0 = time.time()
    for mode in modes:
        key, grid = grids[mode]
        for attack in attacks:
            pts = []
            for value in grid:
                det_kw = {"mode": mode, key: value}
                clean = metrics.summarise(
                    run_many("clean", args.trials, args.seed, path=args.path,
                             duration=args.duration, det_kw=det_kw))
                spoof = metrics.summarise(
                    run_many(attack, args.trials, args.seed, path=args.path,
                             duration=args.duration, det_kw=det_kw))
                pts.append((clean.false_alarm_rate, spoof.detection_rate))
                print("  {:<6} {:<6} {}={:<5} {:7.2f} FA/hour  detect {:5.1f}%".format(
                    mode, attack, key, value,
                    clean.false_alarms_per_hour,
                    100.0 * spoof.detection_rate))
            curves["{} / {}".format(attack, mode)] = metrics.roc_points(pts)

    plots.plot_roc(curves, RESULTS / "roc.png")
    # One false alarm in the whole clean set is the smallest non-zero rate
    # measurable here, so anything below it reads as 0.00 without being zero.
    epochs = args.trials * max(args.duration - 30.0, 1.0)
    floor = 3600.0 / epochs
    print("\nROC -> roc.png   ({:.0f}s, {} detector runs)".format(
        time.time() - t0,
        len(modes) * len(attacks) * 10 * args.trials * 2))
    print("note: with {} trials the resolution floor is {:.1f} FA/hour (one "
          "onset).\n      '0.00' therefore means 'below {:.1f}', not zero -- "
          "raise --trials to resolve it.".format(args.trials, floor, floor))


def cmd_drift_sweep(args) -> None:
    """Find the walk-off rate at which the Phase 0 detector stops working."""
    rates = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0]
    det_kw = dict(mode=args.mode)
    detection, latency = [], []

    print("\nwalk-off sensitivity, {} trials per rate\n".format(args.trials))
    print("{:>8}  {:>8}  {:>9}  {:>10}".format(
        "rate", "detect", "lat-p50", "offset@p50"))
    for r in rates:
        s = metrics.summarise(
            run_many("drift", args.trials, args.seed, path=args.path,
                     duration=args.duration, det_kw=det_kw,
                     spoof_params=dict(rate_mps=r)))
        detection.append(s.detection_rate)
        latency.append(s.latency_median_s)
        offset = ("  --" if s.latency_median_s is None
                  else "{:7.1f} m".format(r * s.latency_median_s))
        lat = ("    -- " if s.latency_median_s is None
               else "{:6.1f}s".format(s.latency_median_s))
        print("{:>6.2f} m/s  {:>7.1f}%  {}  {}".format(
            r, 100.0 * s.detection_rate, lat, offset))

    plots.plot_drift_sweep(rates, detection, latency,
                           RESULTS / "drift_sweep.png")
    print("\n'offset@p50' is how far the attack had already moved the fix by "
          "the time it was caught.\nsweep -> drift_sweep.png")


def cmd_gain_sweep(args) -> None:
    """The central Phase 0 trade-off, measured.

    `vel_gain` sets how readily the estimate adopts GPS's notion of
    velocity. High, and clean flight is quiet but a walk-off is adopted as
    truth and becomes invisible. Low, and the estimate's own velocity error
    grows unchecked. Phase 2 stops choosing this by hand: the Kalman gain
    falls out of the covariance.
    """
    gains = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
    rates = [0.5, 1.0, 2.5]

    print("\nvelocity-anchoring gain vs detectability, {} trials each\n".format(
        args.trials))
    header = "{:>9} {:>11}".format("vel_gain", "FA/hour")
    for r in rates:
        header += "{:>12}".format("drift {}".format(r))
    print(header)

    for vg in gains:
        det_kw = dict(mode=args.mode, vel_gain=vg)
        clean = metrics.summarise(
            run_many("clean", args.trials, args.seed, path=args.path,
                     duration=args.duration, det_kw=det_kw))
        row = "{:>9.3f} {:>10.2f} ".format(vg, clean.false_alarms_per_hour)
        for r in rates:
            s = metrics.summarise(
                run_many("drift", args.trials, args.seed, path=args.path,
                         duration=args.duration, det_kw=det_kw,
                         spoof_params=dict(rate_mps=r)))
            row += "{:>11.0f}%".format(100.0 * s.detection_rate)
        print(row)
    print("\n'drift X' columns are detection rate at a walk-off of X m/s.")


# --------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="eval.run", description="GNSS spoof detector evaluation (Phase 0)")
    p.add_argument("--scenario", default="jump",
                   choices=["clean", "jump", "drift", "replay", "all"])
    p.add_argument("--path", default="survey",
                   choices=["survey", "orbit", "line", "hover"],
                   help="ground-truth flight pattern")
    p.add_argument("--mode", default="cusum",
                   choices=["cusum", "nsigma", "both"],
                   help="decision rule ('both' is ROC only)")
    p.add_argument("--trials", type=int, default=12,
                   help="independent seeds per configuration")
    p.add_argument("--duration", type=float, default=240.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--roc", action="store_true", help="sweep threshold, plot ROC")
    p.add_argument("--roc-attacks", default="jump,drift")
    p.add_argument("--drift-sweep", action="store_true",
                   help="detection vs walk-off rate")
    p.add_argument("--gain-sweep", action="store_true",
                   help="velocity-anchoring gain vs detectability")
    args = p.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    np.set_printoptions(precision=2, suppress=True)

    if args.roc:
        cmd_roc(args)
        print()
        return

    # 'both' only means anything when two curves are being compared.
    if args.mode == "both":
        args.mode = "cusum"
    if args.drift_sweep:
        cmd_drift_sweep(args)
    elif args.gain_sweep:
        cmd_gain_sweep(args)
    else:
        cmd_scenarios(args)
    print()


if __name__ == "__main__":
    main()
