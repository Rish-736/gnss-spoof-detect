"""The adversary: three ways to lie about position.

No RF is transmitted anywhere in this project. The attack is applied to
the GPS position solution in software, which is legal, perfectly
reproducible, and -- importantly -- leaves us holding ground truth, so we
can measure detection latency to the sample. You cannot get that number
from a real over-the-air attack.

What this does NOT model: the signal-layer fingerprints of a real
spoofer (C/N0 and AGC anomalies, carrier-phase discontinuity, loss of
multipath diversity, satellite-geometry inconsistency). Those are what a
receiver like the u-blox M8N looks at in its own `spoofDetState` flag.
This detector works purely at the position-solution layer, which means it
is complementary to those checks, not a replacement for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPOOF_KINDS = ("clean", "jump", "drift", "replay")


@dataclass
class SpoofResult:
    pos: np.ndarray        # (M, 2) the GPS stream as handed to the detector
    label: np.ndarray      # (M,)   bool, True where the fix is a lie
    onset_s: float         # inf for a clean run
    kind: str
    truth_error: np.ndarray  # (M,) metres of induced position error


def _finish(kind, clean, pos, t, onset_s):
    label = t >= onset_s
    return SpoofResult(pos=pos, label=label, onset_s=onset_s, kind=kind,
                       truth_error=np.linalg.norm(pos - clean, axis=1))


def spoof_clean(t, clean, **_):
    """Control case. Everything flagged here is a false alarm."""
    return _finish("clean", clean, clean.copy(), t, np.inf)


def spoof_jump(t, clean, onset_s=60.0, magnitude_m=50.0, bearing_deg=120.0):
    """Teleport the fix. A crude attacker, or a captured-then-released
    receiver reacquiring onto the wrong signal.

    Easy to catch: the full error appears in a single epoch, so any
    sensible threshold trips immediately. Included as the sanity check,
    not as the interesting case.
    """
    b = np.deg2rad(bearing_deg)
    offset = magnitude_m * np.array([np.cos(b), np.sin(b)])
    pos = clean.copy()
    pos[t >= onset_s] += offset
    return _finish("jump", clean, pos, t, onset_s)


def spoof_drift(t, clean, onset_s=60.0, rate_mps=0.5, bearing_deg=120.0,
                max_offset_m=400.0):
    """Walk the fix away at a constant rate. THE case that matters.

    A competent attacker captures the receiver cleanly and then pulls it
    off slowly, because a slow pull is hidden inside the victim's own
    inertial drift. Note what this looks like to a filter that tracks
    velocity: a position ramp is indistinguishable from the aircraft
    genuinely having a slightly different velocity -- unless you have an
    independent velocity reference, which is exactly what the
    accelerometers are. The detector's velocity gain therefore decides
    whether the attacker wins.
    """
    b = np.deg2rad(bearing_deg)
    direction = np.array([np.cos(b), np.sin(b)])
    elapsed = np.clip(t - onset_s, 0.0, None)
    magnitude = np.minimum(rate_mps * elapsed, max_offset_m)
    pos = clean + magnitude[:, None] * direction
    return _finish("drift", clean, pos, t, onset_s)


def spoof_replay(t, clean, onset_s=60.0, lag_s=25.0):
    """Feed back an earlier slice of the aircraft's own track, time-shifted
    and offset so the handover at onset is continuous.

    Nasty because the stream stays internally self-consistent: the
    positions, speeds and turn rates are all ones this airframe really
    flew, so any plausibility check on the GPS stream alone passes. It
    only breaks against an independent witness.
    """
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    lag = int(round(lag_s / dt))
    k0 = int(np.searchsorted(t, onset_s))

    pos = clean.copy()
    src = np.clip(np.arange(k0, t.size) - lag, 0, t.size - 1)
    # Stitch so there is no discontinuity at the moment of capture.
    seam = clean[k0] - clean[max(k0 - lag, 0)]
    pos[k0:] = clean[src] + seam
    return _finish("replay", clean, pos, t, onset_s)


_ATTACKS = {
    "clean": spoof_clean,
    "jump": spoof_jump,
    "drift": spoof_drift,
    "replay": spoof_replay,
}


def apply_spoof(t: np.ndarray, clean_pos: np.ndarray, kind: str = "jump",
                **params) -> SpoofResult:
    """Corrupt a GPS position stream. `t` and `clean_pos` are per-GPS-epoch."""
    if kind not in _ATTACKS:
        raise ValueError(f"unknown spoof {kind!r}; choose from {SPOOF_KINDS}")
    return _ATTACKS[kind](np.asarray(t, float), np.asarray(clean_pos, float),
                          **params)
