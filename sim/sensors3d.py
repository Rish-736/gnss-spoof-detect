"""Phase 1 sensor models: 3-axis IMU, full AHRS attitude, 3-D GNSS.

The error that dominates everything
-----------------------------------
In 2-D we could ignore gravity. In 3-D the accelerometer measures specific
force, so the detector must subtract a 9.81 m/s^2 gravity vector using its
*estimated* attitude. Any attitude error leaks gravity straight into the
horizontal channels:

    horizontal error = 9.81 * sin(tilt error)

    0.1 deg  ->  0.017 m/s^2      (comparable to the accelerometer noise)
    0.5 deg  ->  0.086 m/s^2      (twice the noise)
    1.0 deg  ->  0.171 m/s^2      (four times the noise)
    2.0 deg  ->  0.342 m/s^2      (eight times the noise)

Double-integrated, 1 degree of standing tilt error is ~0.09 m/s^2 of phantom
acceleration, which becomes 77 m of position error in 30 s if nothing corrects
it. That is larger than most of the spoofing attacks we are trying to detect,
which is why the attitude error budget below is the most consequential set of
numbers in this file -- more so than the accelerometer spec.

Yaw error is the gentler one: it misrotates real horizontal acceleration
rather than leaking gravity, and a quadrotor's horizontal specific force is
~0, so a yaw error mostly rotates a small quantity. Tilt is what hurts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attitude import quat_mult, quat_normalize
from .trajectory3d import Trajectory3D

DOWN = 2


@dataclass
class Imu3DSpec:
    """MPU-6050-class 6-DOF IMU, three axes.

    Densities are continuous-time; per-sample sigma is density * sqrt(fs).
    Turn-on bias is drawn once per run and held; bias random walk models the
    slow thermal wander on top of it.

    These are datasheet-order figures. `tools/allan.py` (Phase 1) replaces
    them with coefficients measured from the real part, which is what stops a
    simulation-tuned result from being self-fulfilling.
    """

    accel_noise_density: float = 3.9e-3    # m/s^2/sqrt(Hz)   ~400 ug/sqrt(Hz)
    accel_bias_sigma: float = 0.05         # m/s^2  turn-on, per axis
    accel_bias_rw: float = 5.0e-4          # m/s^2/sqrt(s)
    gyro_noise_density: float = 3.0e-4     # rad/s/sqrt(Hz)
    gyro_bias_sigma: float = 5.0e-4        # rad/s  (~0.03 deg/s, calibrated)
    gyro_bias_rw: float = 1.0e-5           # rad/s/sqrt(s)


@dataclass
class Ahrs3DSpec:
    """Attitude as the autopilot's filter publishes it.

    Split into tilt and heading because they fail differently and matter
    differently. Tilt is observable from the accelerometer itself (gravity is
    a reference vector), so a good AHRS holds it tight -- but every tenth of a
    degree leaks gravity. Heading comes from the magnetometer: looser, and
    sensitive to hard-iron error near motors, but far less damaging.

    Critically, both references (gravity and magnetic field) are *physical and
    local*. Neither can be spoofed from a distance, which is what makes the
    attitude estimate a trustworthy input to a GNSS integrity check.
    """

    tilt_bias_deg: float = 0.3        # standing roll/pitch error, per run
    tilt_wander_deg: float = 0.2      # slow wander on top of it
    tilt_noise_deg: float = 0.05      # per-sample
    yaw_bias_deg: float = 2.0         # magnetometer calibration residual
    yaw_wander_deg: float = 1.0
    yaw_noise_deg: float = 0.3
    wander_period_s: float = 60.0


@dataclass
class Gnss3DSpec:
    """u-blox-class receiver, 3-D position.

    Vertical is deliberately worse: satellite geometry is one-sided (there is
    nothing below the horizon), so VDOP is typically 1.5-2x HDOP. A real
    receiver's error is also correlated over seconds to minutes from
    ionosphere and multipath; modelling it as white is a known optimism,
    recorded in the README.
    """

    rate_hz: float = 1.0
    sigma_horizontal_m: float = 2.5
    sigma_vertical_m: float = 4.0


@dataclass
class Imu3DStream:
    t: np.ndarray           # (N,)
    accel: np.ndarray       # (N, 3) specific force, body frame [m/s^2]
    gyro: np.ndarray        # (N, 3) angular rate, body frame [rad/s]
    accel_bias: np.ndarray  # (N, 3) truth, diagnostics only
    gyro_bias: np.ndarray   # (N, 3) truth, diagnostics only


@dataclass
class Gnss3DStream:
    idx: np.ndarray         # (M,) index into the truth timeline
    t: np.ndarray           # (M,)
    pos: np.ndarray         # (M, 3) NED [m] -- honest fix, pre-attack
    dt: float


def _random_walk(n, step_sigma, rng, axes=3):
    return np.cumsum(rng.normal(0.0, step_sigma, size=(n, axes)), axis=0)


def _slow_wander(t, amplitude, period, rng, axes=3):
    """Smooth low-frequency drift with a random phase per axis."""
    phase = rng.uniform(0.0, 2.0 * np.pi, size=axes)
    return amplitude * np.sin(2.0 * np.pi * t[:, None] / period + phase)


def simulate_imu3d(traj: Trajectory3D, spec: Imu3DSpec | None = None,
                   rng: np.random.Generator | None = None) -> Imu3DStream:
    """Specific force and angular rate as a strapdown IMU would report them."""
    spec = spec or Imu3DSpec()
    rng = rng or np.random.default_rng()
    n, fs = len(traj), traj.fs
    dt = 1.0 / fs

    accel_bias = (rng.normal(0.0, spec.accel_bias_sigma, size=3)
                  + _random_walk(n, spec.accel_bias_rw * np.sqrt(dt), rng))
    gyro_bias = (rng.normal(0.0, spec.gyro_bias_sigma, size=3)
                 + _random_walk(n, spec.gyro_bias_rw * np.sqrt(dt), rng))

    accel = (traj.f_body + accel_bias
             + rng.normal(0.0, spec.accel_noise_density * np.sqrt(fs), size=(n, 3)))
    gyro = (traj.omega + gyro_bias
            + rng.normal(0.0, spec.gyro_noise_density * np.sqrt(fs), size=(n, 3)))

    return Imu3DStream(t=traj.t, accel=accel, gyro=gyro,
                       accel_bias=accel_bias, gyro_bias=gyro_bias)


def simulate_ahrs3d(traj: Trajectory3D, spec: Ahrs3DSpec | None = None,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Attitude estimate, as (N, 4) body-to-nav quaternions.

    The error is applied as a small rotation in the **nav** frame, because
    that is the frame in which it does damage: the horizontal components of
    the error vector are exactly the tilt that leaks gravity, and the vertical
    component is the heading error.
    """
    spec = spec or Ahrs3DSpec()
    rng = rng or np.random.default_rng()
    n, t = len(traj), traj.t

    tilt = (np.deg2rad(rng.normal(0.0, spec.tilt_bias_deg, size=2))
            + _slow_wander(t, np.deg2rad(spec.tilt_wander_deg),
                           spec.wander_period_s, rng, axes=2)
            + rng.normal(0.0, np.deg2rad(spec.tilt_noise_deg), size=(n, 2)))
    yaw = (np.deg2rad(rng.normal(0.0, spec.yaw_bias_deg))
           + _slow_wander(t, np.deg2rad(spec.yaw_wander_deg),
                          spec.wander_period_s * 1.3, rng, axes=1)[:, 0]
           + rng.normal(0.0, np.deg2rad(spec.yaw_noise_deg), size=n))

    err = np.column_stack([tilt, yaw])              # (N, 3) rotation vector, nav
    out = np.empty((n, 4))
    for i in range(n):
        theta = np.linalg.norm(err[i])
        if theta < 1e-12:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = err[i] / theta
            dq = np.concatenate([[np.cos(theta / 2)], np.sin(theta / 2) * axis])
        out[i] = quat_normalize(quat_mult(dq, traj.quat[i]))   # nav-frame error
    return out


def simulate_gnss3d(traj: Trajectory3D, spec: Gnss3DSpec | None = None,
                    rng: np.random.Generator | None = None) -> Gnss3DStream:
    """Honest 3-D fixes at the receiver's update rate."""
    spec = spec or Gnss3DSpec()
    rng = rng or np.random.default_rng()

    stride = int(round(traj.fs / spec.rate_hz))
    idx = np.arange(0, len(traj), stride)
    sigma = np.array([spec.sigma_horizontal_m, spec.sigma_horizontal_m,
                      spec.sigma_vertical_m])
    pos = traj.pos[idx] + rng.normal(0.0, 1.0, size=(idx.size, 3)) * sigma
    return Gnss3DStream(idx=idx, t=traj.t[idx], pos=pos, dt=1.0 / spec.rate_hz)
