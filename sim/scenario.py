"""Assemble truth + sensors + attack into one replayable scenario.

The important thing in this file is `iter_stream()`. It hands out samples
in strict time order and hands out *only* what a real flight computer
would have: body-frame IMU at 100 Hz and a GPS position at 1 Hz. Ground
truth stays in the Scenario object, where only the scoring code touches
it. Keeping that boundary honest is what makes the Phase 3 swap to a live
serial feed a drop-in -- and what stops us writing a detector that
accidentally cheats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from . import sensors, spoofer, trajectory


@dataclass
class Scenario:
    traj: trajectory.Trajectory
    imu: sensors.ImuStream
    gps: sensors.GpsStream
    ahrs: np.ndarray                 # (N,) heading [rad] as the autopilot reports it
    attack: spoofer.SpoofResult
    seed: int
    imu_spec: sensors.ImuSpec = field(default_factory=sensors.ImuSpec)
    gps_spec: sensors.GpsSpec = field(default_factory=sensors.GpsSpec)
    ahrs_spec: sensors.AhrsSpec = field(default_factory=sensors.AhrsSpec)

    # ---- what the detector is allowed to know ----------------------------
    @property
    def fs_imu(self) -> float:
        return self.traj.fs

    @property
    def dt_gps(self) -> float:
        return self.gps.dt

    # ---- ground truth: scoring only --------------------------------------
    @property
    def onset_s(self) -> float:
        return self.attack.onset_s

    @property
    def kind(self) -> str:
        return self.attack.kind

    def iter_stream(self) -> Iterator[tuple]:
        """Yield ('imu', t, accel_body, gyro_z) and ('gps', t, pos) in time
        order, exactly as they would arrive off the wire."""
        gps_at = {int(i): k for k, i in enumerate(self.gps.idx)}
        for i in range(len(self.traj)):
            # GPS first at a shared instant, so the detector's clock reads a
            # whole epoch at each fix and reported latencies come out as exact
            # multiples of the GPS interval rather than offset by one IMU tick.
            k = gps_at.get(i)
            if k is not None:
                yield ("gps", self.gps.t[k], self.attack.pos[k])
            yield ("imu", self.imu.t[i], self.imu.accel[i], self.imu.gyro[i],
                   self.ahrs[i])


def build(spoof: str = "jump", path: str = "survey", duration: float = 240.0,
          fs_imu: float = 100.0, gps_rate_hz: float = 1.0, seed: int = 0,
          imu_spec: sensors.ImuSpec | None = None,
          gps_spec: sensors.GpsSpec | None = None,
          ahrs_spec: sensors.AhrsSpec | None = None,
          path_params: dict | None = None,
          spoof_params: dict | None = None) -> Scenario:
    """Build one scenario. `seed` makes it bit-for-bit reproducible."""
    rng = np.random.default_rng(seed)
    imu_spec = imu_spec or sensors.ImuSpec()
    gps_spec = gps_spec or sensors.GpsSpec(rate_hz=gps_rate_hz)
    ahrs_spec = ahrs_spec or sensors.AhrsSpec()

    traj = trajectory.generate(path, duration=duration, fs=fs_imu,
                               **(path_params or {}))
    imu = sensors.simulate_imu(traj, imu_spec, rng)
    ahrs = sensors.simulate_ahrs(traj, ahrs_spec, rng)
    gps = sensors.simulate_gps(traj, gps_spec, rng)
    attack = spoofer.apply_spoof(gps.t, gps.pos, spoof, **(spoof_params or {}))

    return Scenario(traj=traj, imu=imu, gps=gps, ahrs=ahrs, attack=attack,
                    seed=seed, imu_spec=imu_spec, gps_spec=gps_spec,
                    ahrs_spec=ahrs_spec)


#: Presets used by eval/run.py. Onset sits well clear of the detector's
#: warm-up so we never confuse "not yet calibrated" with "not detected".
PRESETS = {
    "clean":  dict(spoof="clean"),
    "jump":   dict(spoof="jump",   spoof_params=dict(onset_s=80.0, magnitude_m=50.0)),
    "drift":  dict(spoof="drift",  spoof_params=dict(onset_s=80.0, rate_mps=0.5)),
    "replay": dict(spoof="replay", spoof_params=dict(onset_s=80.0, lag_s=25.0)),
}
