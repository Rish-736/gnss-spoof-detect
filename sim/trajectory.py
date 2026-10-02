"""Ground-truth trajectory generation.

Frame convention (Phase 0)
--------------------------
Everything lives in a 2-D local tangent plane with NED-style axes:

    axis 0 = North [m]
    axis 1 = East  [m]

Altitude is held constant. Working in a local plane instead of WGS84
lat/lon keeps the geometry linear; the flat-earth error is well under a
centimetre over the few-hundred-metre flights simulated here, so it buys
simplicity for free. Phase 1 extends this to full 3-D with quaternion
attitude and a proper lat/lon -> ENU conversion.

The trajectory is the *truth*. The detector never sees it -- only the
evaluation code does, and only to score the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

NORTH, EAST = 0, 1

# Below this ground speed the velocity vector is too noisy to define a
# heading from, so we hold the last good one (a real airframe would fall
# back on its magnetometer).
_HEADING_SPEED_FLOOR = 0.5  # m/s


@dataclass
class Trajectory:
    """Noise-free truth for one flight."""

    t: np.ndarray         # (N,)   seconds since start
    pos: np.ndarray       # (N, 2) metres, [north, east]
    vel: np.ndarray       # (N, 2) m/s, nav frame
    acc: np.ndarray       # (N, 2) m/s^2, nav frame
    heading: np.ndarray   # (N,)   rad, 0 = north, positive toward east
    yaw_rate: np.ndarray  # (N,)   rad/s
    fs: float             # Hz
    kind: str

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.vel, axis=1)


# --------------------------------------------------------------------------
# Path shapes. Each returns position only; velocity/accel/heading are
# derived once, consistently, in generate().
# --------------------------------------------------------------------------

def _path_line(t, speed=12.0, bearing_deg=30.0):
    """Straight constant-speed transit."""
    b = np.deg2rad(bearing_deg)
    return np.stack([speed * t * np.cos(b), speed * t * np.sin(b)], axis=1)


def _path_orbit(t, radius=80.0, period=55.0):
    """Circular loiter -- what a surveillance UAV does over a point of interest."""
    w = 2.0 * np.pi / period
    return np.stack([radius * np.cos(w * t), radius * np.sin(w * t)], axis=1)


def _path_survey(t, leg=180.0, spacing=45.0, speed=12.0, n_legs=12):
    """Lawnmower / boustrophedon mapping pattern, flown at constant speed.

    Built as a waypoint polyline, then sampled by arc length so the speed
    is genuinely constant along the path (including round the corners,
    once smoothing is applied).
    """
    wps = [(0.0, 0.0)]
    east = 0.0
    for i in range(n_legs):
        north = leg if i % 2 == 0 else 0.0
        wps.append((north, east))
        east += spacing
        wps.append((north, east))
    wps = np.asarray(wps, float)

    seg_len = np.linalg.norm(np.diff(wps, axis=0), axis=1)
    s_wp = np.concatenate([[0.0], np.cumsum(seg_len)])
    s = np.clip(speed * t, 0.0, s_wp[-1])
    return np.stack([np.interp(s, s_wp, wps[:, 0]),
                     np.interp(s, s_wp, wps[:, 1])], axis=1)


def _path_hover(t, amplitude_m=0.7, period=11.0):
    """Station-keeping. Not white noise -- a slow wind-driven wander, which
    is what a hovering multirotor's position error actually looks like."""
    return np.stack([amplitude_m * np.sin(2.0 * np.pi * t / period),
                     amplitude_m * np.cos(2.0 * np.pi * t / (period * 1.37))],
                    axis=1)


_PATHS = {
    "line": _path_line,
    "orbit": _path_orbit,
    "survey": _path_survey,
    "hover": _path_hover,
}

PATH_KINDS = tuple(_PATHS)


# --------------------------------------------------------------------------

def _central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Second-order central difference, one-sided at the ends."""
    d = np.empty_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    d[0] = (x[1] - x[0]) / dt
    d[-1] = (x[-1] - x[-2]) / dt
    return d


def _heading_from_velocity(vel: np.ndarray, fs: float) -> np.ndarray:
    """Course over ground, holding through low-speed gaps, then unwrapped."""
    speed = np.linalg.norm(vel, axis=1)
    psi = np.arctan2(vel[:, EAST], vel[:, NORTH])
    good = speed > _HEADING_SPEED_FLOOR

    if not good.any():
        return np.zeros_like(speed)

    # Forward-fill the last valid heading across the low-speed gaps.
    idx = np.where(good, np.arange(psi.size), 0)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(good))
    idx[:first] = first
    psi = np.unwrap(psi[idx])

    # A real airframe cannot snap its heading; take the edge off the
    # forward-fill seams so the yaw rate stays physical.
    return gaussian_filter1d(psi, max(0.2 * fs, 1.0), mode="nearest")


def generate(kind: str = "survey", duration: float = 180.0, fs: float = 100.0,
             corner_smooth_s: float = 1.2, **params) -> Trajectory:
    """Build one ground-truth flight.

    Parameters
    ----------
    kind : one of PATH_KINDS
    duration : flight length [s]
    fs : IMU/truth sample rate [Hz]
    corner_smooth_s : low-pass width applied to position before
        differentiating. A raw waypoint polyline has corners, and a corner
        demands infinite acceleration; smoothing rounds them into turns a
        real airframe could fly.
    """
    if kind not in _PATHS:
        raise ValueError(f"unknown trajectory {kind!r}; choose from {PATH_KINDS}")

    dt = 1.0 / fs
    t = np.arange(0.0, duration, dt)
    pos = _PATHS[kind](t, **params).astype(float)

    if corner_smooth_s > 0.0:
        sigma = corner_smooth_s * fs
        pos = np.stack([gaussian_filter1d(pos[:, i], sigma, mode="nearest")
                        for i in (NORTH, EAST)], axis=1)

    vel = _central_diff(pos, dt)
    acc = _central_diff(vel, dt)
    heading = _heading_from_velocity(vel, fs)
    yaw_rate = _central_diff(heading, dt)

    return Trajectory(t=t, pos=pos, vel=vel, acc=acc, heading=heading,
                      yaw_rate=yaw_rate, fs=fs, kind=kind)
