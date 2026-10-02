"""Scoring. This is the file that turns "it works" into a number.

Definitions used throughout, chosen so they cannot flatter the detector:

detection rate
    Fraction of spoofed *runs* caught. A run only counts as caught if the
    detector did not already false-alarm before the attack started --
    otherwise a detector that alarms constantly would score 100%.

false-alarm rate
    Alarm *onsets* per eligible epoch, measured on clean flight only, and
    also reported as alarms per hour because that is the unit an operator
    actually cares about.

    Onsets, not flagged epochs. The detector latches, so one false alarm at
    epoch 40 of a 210-epoch run would score as a 81% "false-alarm rate" if
    flagged epochs were counted -- which says nothing about how often the
    detector cries wolf, only that it stays latched afterwards. Counting
    rising edges makes the number mean what it claims to mean, and keeps it
    comparable across flight lengths.

detection latency
    Seconds from attack onset to the first flag. The number that actually
    matters operationally: at 15 m/s, every second of latency is 15 m of
    the aircraft's position you can no longer trust.

Warm-up epochs are excluded from every count. A detector that has not yet
learned its own residual statistics is not yet on duty, and scoring it
there would measure the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunScore:
    kind: str
    detected: bool
    latency_s: float | None
    n_eligible_clean: int      # epochs that could have false-alarmed
    n_false_onsets: int        # alarm onsets (rising edges) among them
    dt_gps: float
    peak_truth_error_m: float  # how far the attack actually moved the fix

    @property
    def false_alarmed(self) -> bool:
        return self.n_false_onsets > 0


def _rising(flags: np.ndarray) -> np.ndarray:
    """True where a flag turns on, so a latched alarm counts once."""
    prev = np.concatenate([[False], flags[:-1]])
    return flags & ~prev


def score_run(t_gps, flags, onset_s, warmup_fixes, kind="?",
              truth_error=None, dt_gps=1.0) -> RunScore:
    """Score one run of one scenario."""
    t = np.asarray(t_gps, float)
    flags = np.asarray(flags, bool)
    eligible = np.arange(t.size) > warmup_fixes

    pre = eligible & (t < onset_s)
    post = eligible & (t >= onset_s)

    n_false = int(np.count_nonzero(_rising(flags) & pre))
    hit = flags & post
    detected = bool(hit.any())
    latency = float(t[hit][0] - onset_s) if detected else None

    peak = 0.0 if truth_error is None else float(np.max(np.abs(truth_error)))
    return RunScore(kind=kind, detected=detected, latency_s=latency,
                    n_eligible_clean=int(pre.sum()), n_false_onsets=n_false,
                    dt_gps=float(dt_gps), peak_truth_error_m=peak)


@dataclass
class Summary:
    kind: str
    n_runs: int
    detection_rate: float          # fraction of runs caught cleanly
    false_alarm_rate: float        # alarm onsets per epoch, clean portion
    false_alarms_per_hour: float   # the same number, in operator units
    clean_minutes: float           # how much clean flight it was measured over
    latency_median_s: float | None
    latency_p90_s: float | None
    latency_worst_s: float | None

    def row(self) -> str:
        def fmt(v):
            return "   --  " if v is None else "{:6.2f} ".format(v)
        return "{:<8} {:>5}  {:>7.1f}%  {:>9.2f}  {} {} {}".format(
            self.kind, self.n_runs, 100.0 * self.detection_rate,
            self.false_alarms_per_hour, fmt(self.latency_median_s),
            fmt(self.latency_p90_s), fmt(self.latency_worst_s))


HEADER = ("{:<8} {:>5}  {:>8}  {:>9}  {:>7} {:>7} {:>7}".format(
    "spoof", "runs", "detect", "FA/hour", "lat-p50", "lat-p90", "lat-max"))


def summarise(scores) -> Summary:
    """Aggregate many runs of the same scenario into reportable numbers."""
    scores = list(scores)
    if not scores:
        raise ValueError("nothing to summarise")
    kind = scores[0].kind
    n = len(scores)

    is_clean = kind == "clean"
    # A detection only counts if the run was not already false-alarming.
    caught = [s for s in scores if s.detected and not s.false_alarmed]
    detection_rate = 0.0 if is_clean else len(caught) / n

    total_epochs = sum(s.n_eligible_clean for s in scores)
    total_false = sum(s.n_false_onsets for s in scores)
    far = total_false / total_epochs if total_epochs else 0.0
    dt = scores[0].dt_gps
    clean_minutes = total_epochs * dt / 60.0
    per_hour = far * 3600.0 / dt if dt > 0 else 0.0

    lat = np.array([s.latency_s for s in caught if s.latency_s is not None])
    if lat.size:
        p50 = float(np.median(lat))
        p90 = float(np.percentile(lat, 90))
        worst = float(lat.max())
    else:
        p50 = p90 = worst = None

    return Summary(kind=kind, n_runs=n, detection_rate=detection_rate,
                   false_alarm_rate=far, false_alarms_per_hour=per_hour,
                   clean_minutes=clean_minutes, latency_median_s=p50,
                   latency_p90_s=p90, latency_worst_s=worst)


def roc_points(far_tpr):
    """Sort (far, tpr) pairs into a monotone curve and report AUC.

    AUC is computed over the observed FAR range only, normalised by it, so
    it is an honest partial AUC rather than one padded out to FAR = 1.
    """
    pts = sorted(far_tpr)
    far = np.array([p[0] for p in pts], float)
    tpr = np.array([p[1] for p in pts], float)
    if far.size < 2 or far[-1] <= far[0]:
        return far, tpr, float("nan")
    auc = float(np.trapezoid(tpr, far) / (far[-1] - far[0]))
    return far, tpr, auc
