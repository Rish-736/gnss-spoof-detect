"""Phase 1: full 3-D ground-truth flight in NED, with attitude.

What changes from Phase 0's 2-D model, and why each change matters
------------------------------------------------------------------
* **Three axes.** Altitude is now a real state. Gravity lives on it, and GPS
  is roughly 1.5x worse vertically than horizontally, so the vertical channel
  is both the noisiest and the one carrying 9.81 m/s^2 of bias-like signal.

* **Attitude is derived, not assumed.** A quadrotor accelerates by tilting, so
  the attitude is fixed by the trajectory (differential flatness). We no
  longer get to pretend the IMU is conveniently aligned with north.

* **The accelerometer measures specific force.** `f_body = R^T (a_nav - g_nav)`.
  For a multirotor that is approximately [0, 0, -T/m]: nearly all of it on the
  body z-axis, almost nothing on the horizontal axes. Horizontal motion is
  encoded in the *attitude*, not in the horizontal accelerometer channels --
  which is precisely why Phase 1 is a fight about tilt accuracy.

Phase 0 kept its own 2-D module so its published numbers stay reproducible.
This file is the Phase 1 replacement, not an edit of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .attitude import (G_NED, attitude_from_accel_yaw, quat_to_euler,
                       quat_to_rot, rates_from_attitude)

NORTH, EAST, DOWN = 0, 1, 2
_YAW_SPEED_FLOOR = 0.5   # m/s below which course over ground is meaningless


@dataclass
class Trajectory3D:
    """Noise-free truth for one 3-D flight."""

    t: np.ndarray        # (N,)   s
    pos: np.ndarray      # (N, 3) m, NED (z is DOWN, so altitude = -z)
    vel: np.ndarray      # (N, 3) m/s, nav frame
    acc: np.ndarray      # (N, 3) m/s^2, nav frame
    quat: np.ndarray     # (N, 4) body->nav, scalar first
    omega: np.ndarray    # (N, 3) rad/s, body frame
    f_body: np.ndarray   # (N, 3) m/s^2, specific force in body frame
    fs: float
    kind: str

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    @property
    def altitude(self) -> np.ndarray:
        """Height above the start plane [m], positive up."""
        return -self.pos[:, DOWN]

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.vel, axis=1)

    @property
    def euler(self) -> np.ndarray:
        """(N, 3) roll/pitch/yaw [rad] -- for plots and diagnostics only."""
        return np.array([quat_to_euler(q) for q in self.quat])

    @property
    def tilt_deg(self) -> np.ndarray:
        """Angle between body-z and vertical [deg]: how hard it is manoeuvring."""
        c = np.clip([quat_to_rot(q)[2, 2] for q in self.quat], -1.0, 1.0)
        return np.degrees(np.arccos(c))


# ---------------------------------------------------------------------------
# path shapes -- position only; everything else is derived consistently
# ---------------------------------------------------------------------------

def _path_survey(t, leg=180.0, spacing=45.0, speed=12.0, n_legs=12,
                 altitude=60.0):
    """Lawnmower mapping pattern at constant altitude."""
    wps, east = [(0.0, 0.0)], 0.0
    for i in range(n_legs):
        north = leg if i % 2 == 0 else 0.0
        wps.append((north, east))
        east += spacing
        wps.append((north, east))
    wps = np.asarray(wps, float)

    seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
    s_wp = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.clip(speed * t, 0.0, s_wp[-1])
    return np.stack([np.interp(s, s_wp, wps[:, 0]),
                     np.interp(s, s_wp, wps[:, 1]),
                     np.full_like(t, -altitude)], axis=1)


def _path_orbit(t, radius=80.0, period=55.0, altitude=80.0):
    """Circular loiter -- what a surveillance UAV does over a point."""
    w = 2.0 * np.pi / period
    return np.stack([radius * np.cos(w * t),
                     radius * np.sin(w * t),
                     np.full_like(t, -altitude)], axis=1)


def _path_helix(t, radius=70.0, period=50.0, climb_rate=1.2, altitude=40.0,
                max_altitude=160.0):
    """Orbit with a steady climb.

    Included because it is the one profile that genuinely exercises the
    vertical channel: a sustained climb is a sustained non-zero vertical
    acceleration demand, which is where a vertical accelerometer bias and a
    gravity-compensation error become hard to tell apart.
    """
    w = 2.0 * np.pi / period
    alt = np.minimum(altitude + climb_rate * t, max_altitude)
    return np.stack([radius * np.cos(w * t),
                     radius * np.sin(w * t),
                     -alt], axis=1)


def _path_hover(t, amplitude=0.7, period=11.0, altitude=30.0):
    """Station-keeping: slow wind-driven wander, not white noise."""
    return np.stack([amplitude * np.sin(2.0 * np.pi * t / period),
                     amplitude * np.cos(2.0 * np.pi * t / (period * 1.37)),
                     -altitude + 0.3 * np.sin(2.0 * np.pi * t / 17.0)], axis=1)


def _path_transit(t, speed=14.0, bearing_deg=35.0, altitude=100.0):
    """Straight-line cruise -- the benign case, useful as a control."""
    b = np.deg2rad(bearing_deg)
    return np.stack([speed * t * np.cos(b),
                     speed * t * np.sin(b),
                     np.full_like(t, -altitude)], axis=1)


_PATHS = {
    "survey": _path_survey,
    "orbit": _path_orbit,
    "helix": _path_helix,
    "hover": _path_hover,
    "transit": _path_transit,
}

PATH_KINDS = tuple(_PATHS)


# ---------------------------------------------------------------------------

def _central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    d = np.empty_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    d[0] = (x[1] - x[0]) / dt
    d[-1] = (x[-1] - x[-2]) / dt
    return d


def _yaw_from_velocity(vel: np.ndarray, fs: float) -> np.ndarray:
    """Point the nose along the ground track, holding through low-speed gaps."""
    ground = np.linalg.norm(vel[:, :2], axis=1)
    yaw = np.arctan2(vel[:, EAST], vel[:, NORTH])
    good = ground > _YAW_SPEED_FLOOR
    if not good.any():
        return np.zeros(len(vel))
    idx = np.where(good, np.arange(yaw.size), 0)
    np.maximum.accumulate(idx, out=idx)
    idx[:int(np.argmax(good))] = int(np.argmax(good))
    return gaussian_filter1d(np.unwrap(yaw[idx]), max(0.2 * fs, 1.0),
                             mode="nearest")


def generate(kind: str = "survey", duration: float = 240.0, fs: float = 100.0,
             corner_smooth_s: float = 1.2, **params) -> Trajectory3D:
    """Build one 3-D ground-truth flight, attitude and specific force included.

    Order matters: position is smoothed *first*, then everything else is
    derived from the smoothed path, so the kinematics stay mutually consistent.
    Phase 0 taught us that truth generated one way and consumed another leaves
    a systematic error indistinguishable from sensor bias.
    """
    if kind not in _PATHS:
        raise ValueError("unknown trajectory %r; choose from %s" % (kind, PATH_KINDS))

    dt = 1.0 / fs
    t = np.arange(0.0, duration, dt)
    pos = _PATHS[kind](t, **params).astype(float)

    # A waypoint polyline has corners, and a corner demands infinite
    # acceleration -- and in 3-D, infinite acceleration demands a 90 degree
    # bank. Rounding them keeps the implied attitude physical.
    if corner_smooth_s > 0.0:
        sigma = corner_smooth_s * fs
        pos = np.stack([gaussian_filter1d(pos[:, i], sigma, mode="nearest")
                        for i in range(3)], axis=1)

    vel = _central_diff(pos, dt)
    acc = _central_diff(vel, dt)
    yaw = _yaw_from_velocity(vel, fs)

    # Attitude is forced by the acceleration the trajectory demands.
    quat = np.array([attitude_from_accel_yaw(acc[i], yaw[i]) for i in range(len(t))])
    omega = rates_from_attitude(quat, dt)

    # What the accelerometer actually senses.
    f_body = np.empty_like(acc)
    for i in range(len(t)):
        f_body[i] = quat_to_rot(quat[i]).T @ (acc[i] - G_NED)

    return Trajectory3D(t=t, pos=pos, vel=vel, acc=acc, quat=quat, omega=omega,
                        f_body=f_body, fs=fs, kind=kind)
