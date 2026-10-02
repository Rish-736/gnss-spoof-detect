"""Sensor error models: what the detector is actually allowed to see.

The whole project rests on an asymmetry between these two sensors:

  GPS  -- error is BOUNDED (a few metres, forever) but the signal arrives
          from 20 000 km away at about 20 W, so an attacker can overpower
          it. Forgeable.
  IMU  -- error GROWS without bound (white accelerometer noise integrated
          twice goes as t^1.5; a bias goes as t^2) but it measures the
          airframe's own physics. To forge it you would have to physically
          push the aircraft. Not forgeable.

So the IMU is an unforgeable but short-sighted witness, and the detector's
job is to use it inside the window where it is still trustworthy.

Noise figures below are datasheet-order values for an MPU-6050 and a
u-blox NEO-M8N. Phase 1 replaces the IMU numbers with coefficients
*measured* from the real part via an Allan-variance run, which is what
stops the simulation from being tuned to succeed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trajectory import EAST, NORTH, Trajectory


@dataclass
class ImuSpec:
    """MPU-6050-class 6-DOF IMU.

    `*_noise_density` are continuous-time densities; the per-sample sigma
    is density * sqrt(fs), which is why a faster IMU is noisier per sample
    but no worse once integrated.

    `*_bias_sigma` is turn-on bias: fixed for a given power cycle, redrawn
    every run.

    The gyro bias is the small one on purpose. Every real airframe averages
    its gyros for a few seconds while sitting still on the ground before
    arming, which removes the turn-on bias down to the part's in-run
    stability -- order 0.03 deg/s for an MPU-6050. That matters enormously
    here: gyro bias tilts the heading estimate, a tilted heading misrotates
    the accelerometers, and the misrotation integrates straight into
    position. Skip the calibration and heading error alone will drift ~25
    degrees over a 4-minute flight, which swamps every other error and makes
    spoof detection impossible. The accelerometer bias is deliberately NOT
    assumed calibrated, because it is observable in flight and the detector
    has to cope with it.
    """

    accel_noise_density: float = 3.9e-3   # m/s^2/sqrt(Hz)  (~400 ug/sqrt(Hz))
    accel_bias_sigma: float = 0.05        # m/s^2   turn-on bias
    accel_bias_rw: float = 5.0e-4         # m/s^2/sqrt(s)   bias instability
    gyro_noise_density: float = 3.0e-4    # rad/s/sqrt(Hz)
    gyro_bias_sigma: float = 5.0e-4       # rad/s   (~0.03 deg/s, post-calibration)
    gyro_bias_rw: float = 1.0e-5          # rad/s/sqrt(s)


@dataclass
class AhrsSpec:
    """Heading from the autopilot's attitude filter.

    On a real airframe the flight controller already fuses gyro,
    accelerometer and magnetometer into attitude (Madgwick, Mahony, or the
    EKF inside PX4/ArduPilot). A spoof detector consumes that estimate
    rather than re-deriving it, so that is what we model.

    The property that matters: this heading comes from a magnetometer, so it
    is INDEPENDENT of GNSS. Deriving heading from GPS course over ground
    instead would hand the attacker the very reference being used to audit
    it -- and in a 2-D model it is not even observable without GPS. Keeping
    the attitude reference out of the attacker's reach is a design
    requirement, not a convenience.

    Error budget is for a calibrated magnetometer on a clean airframe.
    Hard-iron error near motors or payload can be several times worse;
    Phase 3 should measure it on the real vehicle rather than assume it.
    """

    bias_deg: float = 2.0            # per-run offset (calibration residual)
    wander_deg: float = 1.0          # slow wander amplitude
    wander_period_s: float = 60.0
    noise_deg: float = 0.5           # per-sample


@dataclass
class GpsSpec:
    """u-blox-class single-frequency receiver, horizontal position only."""

    rate_hz: float = 1.0
    sigma_m: float = 2.5    # per-axis 1-sigma horizontal position noise


@dataclass
class ImuStream:
    t: np.ndarray          # (N,)
    accel: np.ndarray      # (N, 2) m/s^2 in the BODY frame
    gyro: np.ndarray       # (N,)   rad/s yaw rate about body z
    accel_bias: np.ndarray # (N, 2) truth, for diagnostics only
    gyro_bias: np.ndarray  # (N,)   truth, for diagnostics only


@dataclass
class GpsStream:
    idx: np.ndarray        # (M,) index into the IMU/truth timeline
    t: np.ndarray          # (M,) seconds
    pos: np.ndarray        # (M, 2) metres -- the honest fix, pre-attack
    dt: float              # nominal interval [s]


def _random_walk(n: int, step_sigma: float, rng: np.random.Generator,
                 shape: tuple = ()) -> np.ndarray:
    """Integrated white noise: sigma grows as sqrt(t)."""
    return np.cumsum(rng.normal(0.0, step_sigma, size=(n, *shape)), axis=0)


def simulate_imu(traj: Trajectory, spec: ImuSpec | None = None,
                 rng: np.random.Generator | None = None) -> ImuStream:
    """Turn truth into what a strapdown IMU would actually report.

    Two things happen here, and both matter:

    1. Acceleration is rotated from the navigation frame into the BODY
       frame, because that is the only frame an IMU can measure in. The
       detector has to undo this using its own heading estimate -- and any
       heading error turns into an acceleration error. That coupling is
       the main reason dead-reckoning is hard.
    2. Bias and white noise are added. Bias is the dangerous one: white
       noise averages down, bias integrates.

    Gravity is absent because a level 2-D model puts it entirely on body-z,
    which we do not carry. In the Phase 1 3-D model gravity must be
    subtracted, and attitude error leaking gravity into the horizontal
    axes becomes the dominant dead-reckoning error -- a 1 degree tilt
    error injects 0.17 m/s^2, which is four times the accel noise here.
    """
    spec = spec or ImuSpec()
    rng = rng or np.random.default_rng()
    n = len(traj)
    fs = traj.fs
    dt = 1.0 / fs

    # Nav -> body is the transpose of body -> nav.
    c, s = np.cos(traj.heading), np.sin(traj.heading)
    a_n, a_e = traj.acc[:, NORTH], traj.acc[:, EAST]
    accel_body = np.stack([c * a_n + s * a_e,
                           -s * a_n + c * a_e], axis=1)

    accel_bias = (rng.normal(0.0, spec.accel_bias_sigma, size=2)
                  + _random_walk(n, spec.accel_bias_rw * np.sqrt(dt), rng, (2,)))
    gyro_bias = (rng.normal(0.0, spec.gyro_bias_sigma)
                 + _random_walk(n, spec.gyro_bias_rw * np.sqrt(dt), rng))

    accel_sigma = spec.accel_noise_density * np.sqrt(fs)
    gyro_sigma = spec.gyro_noise_density * np.sqrt(fs)

    accel = accel_body + accel_bias + rng.normal(0.0, accel_sigma, size=(n, 2))
    gyro = traj.yaw_rate + gyro_bias + rng.normal(0.0, gyro_sigma, size=n)

    return ImuStream(t=traj.t, accel=accel, gyro=gyro,
                     accel_bias=accel_bias, gyro_bias=gyro_bias)


def simulate_ahrs(traj: Trajectory, spec: AhrsSpec | None = None,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Heading estimate as the autopilot would publish it, in radians."""
    spec = spec or AhrsSpec()
    rng = rng or np.random.default_rng()

    bias = np.deg2rad(rng.normal(0.0, spec.bias_deg))
    phase = rng.uniform(0.0, 2.0 * np.pi)
    wander = np.deg2rad(spec.wander_deg) * np.sin(
        2.0 * np.pi * traj.t / spec.wander_period_s + phase)
    noise = rng.normal(0.0, np.deg2rad(spec.noise_deg), size=len(traj))
    return traj.heading + bias + wander + noise


def simulate_gps(traj: Trajectory, spec: GpsSpec | None = None,
                 rng: np.random.Generator | None = None) -> GpsStream:
    """Honest GPS fixes at the receiver's update rate.

    White noise only. Real receivers have second-to-minute correlated
    error from ionosphere and multipath, which makes a slow spoof harder
    to separate from a bad sky view -- noted as a Phase 1 limitation.
    """
    spec = spec or GpsSpec()
    rng = rng or np.random.default_rng()

    stride = int(round(traj.fs / spec.rate_hz))
    idx = np.arange(0, len(traj), stride)
    pos = traj.pos[idx] + rng.normal(0.0, spec.sigma_m, size=(idx.size, 2))

    return GpsStream(idx=idx, t=traj.t[idx], pos=pos, dt=1.0 / spec.rate_hz)
