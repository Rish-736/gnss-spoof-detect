"""Figures. Agg backend so this runs headless and in CI."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

TRUE_C = "#1b1b1b"
GPS_C = "#d1495b"
DR_C = "#1f7a8c"
ALARM_C = "#d1495b"
ONSET_C = "#8d99ae"


def _clip(ax, series, cap):
    """Limit the y-axis to `cap` when the series blows past it, and return a
    title suffix saying so. Silent clipping would be a lie; autoscaling to a
    post-alarm runaway would hide the decision."""
    top = float(np.nanmax(series)) if len(series) else 0.0
    if top <= cap:
        return ""
    ax.set_ylim(min(0.0, float(np.nanmin(series))), cap)
    return "   (clipped, peaks at {:.0f})".format(top)


def _mark_events(ax, onset_s, flag_t):
    if np.isfinite(onset_s):
        ax.axvline(onset_s, color=ONSET_C, ls="--", lw=1.2,
                   label="spoof onset")
    if flag_t is not None:
        ax.axvline(flag_t, color=ALARM_C, ls="-", lw=1.4, label="detection")


def plot_scenario(scn, log, out_path, title=None):
    """Four-panel diagnostic for a single run.

    Plan view, then the three signals that make the decision: raw residual,
    its normalised form, and the accumulated evidence.
    """
    t = log["t_gps"]
    flag_t = log["flag_time"]
    onset = scn.onset_s

    fig = plt.figure(figsize=(13.5, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_res = fig.add_subplot(gs[0, 1])
    ax_z = fig.add_subplot(gs[1, 1], sharex=ax_res)
    ax_cu = fig.add_subplot(gs[2, 1], sharex=ax_res)

    # -- plan view (East right, North up: standard navigation convention) --
    truth = scn.traj.pos
    ax_map.plot(truth[:, 1], truth[:, 0], color=TRUE_C, lw=1.6,
                label="true path")
    ax_map.plot(scn.attack.pos[:, 1], scn.attack.pos[:, 0], ".", ms=3.2,
                color=GPS_C, alpha=0.75, label="GPS as received")
    dr = log["dr_pos"]
    ax_map.plot(dr[:, 1], dr[:, 0], color=DR_C, lw=1.1, alpha=0.9,
                label="detector dead-reckoning")

    if np.isfinite(onset):
        k = int(np.searchsorted(t, onset))
        if k < t.size:
            ax_map.plot(scn.attack.pos[k, 1], scn.attack.pos[k, 0], "o",
                        ms=9, mfc="none", mec=ONSET_C, mew=1.8,
                        label="spoof onset")
    if flag_t is not None:
        k = int(np.searchsorted(t, flag_t))
        if k < t.size:
            ax_map.plot(scn.attack.pos[k, 1], scn.attack.pos[k, 0], "X",
                        ms=11, color=ALARM_C, label="detection")

    ax_map.set_xlabel("East [m]")
    ax_map.set_ylabel("North [m]")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.grid(alpha=0.25)
    ax_map.legend(loc="best", fontsize=8.5, framealpha=0.9)
    ax_map.set_title("plan view", fontsize=10)

    # -- raw residual, with the attack's true error for reference ----------
    ax_res.plot(t, log["residual_m"], color=DR_C, lw=1.2,
                label="|GPS - dead-reckoning|")
    ax_res.plot(t, scn.attack.truth_error, color=GPS_C, lw=1.0, ls=":",
                label="true induced error")
    ax_res.set_ylabel("metres")
    ax_res.grid(alpha=0.25)
    ax_res.legend(fontsize=8, loc="upper left")
    _mark_events(ax_res, onset, flag_t)
    ax_res.set_title("residual", fontsize=10)

    # -- normalised residual ----------------------------------------------
    # After latching, the estimate coasts on inertial alone and the residual
    # runs away by design. Autoscaling to that flattens everything that
    # actually mattered, so both decision panels are clipped to the region
    # around the threshold and the clipping is stated.
    ax_z.plot(t, log["z"], color=DR_C, lw=1.2)
    ax_z.axhline(log["cusum_k"], color="#6c757d", ls=":", lw=1.1,
                 label="CUSUM slack k")
    ax_z.set_ylabel("sigma")
    ax_z.grid(alpha=0.25)
    ax_z.legend(fontsize=8, loc="upper left")
    _mark_events(ax_z, onset, flag_t)
    ax_z.set_title("normalised residual z" + _clip(ax_z, log["z"], 10.0),
                   fontsize=10)

    # -- accumulated evidence ---------------------------------------------
    ax_cu.plot(t, log["score"], color=DR_C, lw=1.3)
    ax_cu.axhline(log["threshold"], color=ALARM_C, ls="--", lw=1.2,
                  label="alarm threshold")
    ax_cu.set_ylabel("evidence")
    ax_cu.set_xlabel("time [s]")
    ax_cu.grid(alpha=0.25)
    ax_cu.legend(fontsize=8, loc="upper left")
    _mark_events(ax_cu, onset, flag_t)
    ax_cu.set_title("accumulated evidence ({})".format(log["mode"])
                    + _clip(ax_cu, log["score"], 2.0 * log["threshold"]),
                    fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12.5)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_roc(curves, out_path, title="Detection vs false alarm"):
    """curves: {label: (far_array, tpr_array, auc)}"""
    fig, ax = plt.subplots(figsize=(6.6, 5.4), constrained_layout=True)
    for label, (far, tpr, auc) in curves.items():
        lbl = label if not np.isfinite(auc) else "{}  (pAUC {:.3f})".format(label, auc)
        ax.plot(3600.0 * far, 100.0 * tpr, "o-", ms=4.5, lw=1.5, label=lbl)
    ax.set_xlabel("false alarms per hour of clean flight")
    ax.set_ylabel("detection rate [% of runs]")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(title, fontsize=11)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_latency(summaries, out_path, title="Detection latency by attack"):
    """summaries: list of metrics.Summary (clean runs are skipped)."""
    rows = [s for s in summaries
            if s.kind != "clean" and s.latency_median_s is not None]
    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)

    if not rows:
        ax.text(0.5, 0.5, "no detections to plot", ha="center", va="center")
        ax.axis("off")
    else:
        x = np.arange(len(rows))
        p50 = [s.latency_median_s for s in rows]
        p90 = [s.latency_p90_s for s in rows]
        ax.bar(x - 0.19, p50, 0.38, label="median", color=DR_C)
        ax.bar(x + 0.19, p90, 0.38, label="90th percentile", color=GPS_C,
               alpha=0.85)
        for xi, s in zip(x, rows):
            ax.annotate("{:.0f}%".format(100.0 * s.detection_rate),
                        (xi, max(s.latency_p90_s, s.latency_median_s)),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([s.kind for s in rows])
        ax.set_ylabel("latency [s]")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9)
    ax.set_title(title + "   (label = detection rate)", fontsize=11)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_drift_sweep(rates, detection, latency, out_path):
    """Where the Phase 0 detector gives out, as a function of walk-off rate.

    This is the most informative Phase 0 figure: it locates the boundary
    the EKF work in Phase 2 has to push down.
    """
    fig, ax1 = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    ax1.plot(rates, 100.0 * np.asarray(detection), "o-", color=DR_C, lw=1.6,
             label="detection rate")
    ax1.set_xlabel("walk-off rate [m/s]")
    ax1.set_ylabel("detection rate [%]", color=DR_C)
    ax1.tick_params(axis="y", labelcolor=DR_C)
    ax1.set_ylim(-3, 103)
    ax1.grid(alpha=0.3)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    lat = [np.nan if v is None else v for v in latency]
    ax2.plot(rates, lat, "s--", color=GPS_C, lw=1.4, label="median latency")
    ax2.set_ylabel("median latency [s]", color=GPS_C)
    ax2.tick_params(axis="y", labelcolor=GPS_C)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="center right")
    ax1.set_title("Phase 0 sensitivity limit: slow walk-off", fontsize=11)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
