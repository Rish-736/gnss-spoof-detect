"""Smoke tests. Run with `python -m tests.test_smoke` or `pytest`.

These exist to protect the properties that are easy to break silently:
reproducibility, the truth boundary, and the spec's public API.
"""

from __future__ import annotations

import numpy as np

from detector import SpoofDetector
from eval.run import run_detector, trial
from sim import scenario as scenario_mod
from sim import sensors, trajectory


def test_trajectory_kinematics_are_self_consistent():
    """Integrating the derived velocity must reproduce the position, or every
    downstream comparison is measuring the generator's own error."""
    traj = trajectory.generate("orbit", duration=60.0, fs=100.0)
    dt = 1.0 / traj.fs
    integrated = traj.pos[0] + np.cumsum(traj.vel, axis=0) * dt
    drift = np.linalg.norm(integrated[:-1] - traj.pos[1:], axis=1)
    assert drift.max() < 0.5, "trajectory velocity disagrees with its position"


def test_scenario_is_reproducible():
    a = scenario_mod.build(spoof="drift", seed=7)
    b = scenario_mod.build(spoof="drift", seed=7)
    assert np.array_equal(a.imu.accel, b.imu.accel)
    assert np.array_equal(a.attack.pos, b.attack.pos)
    c = scenario_mod.build(spoof="drift", seed=8)
    assert not np.array_equal(a.imu.accel, c.imu.accel)


def test_stream_never_leaks_ground_truth():
    """The detector's inputs must be reachable from the sensor streams alone.

    A detector that can see `true_pos` would score beautifully and mean
    nothing, so the stream is checked rather than trusted.
    """
    scn = scenario_mod.build(spoof="jump", seed=0, duration=30.0)
    for kind, _t, *payload in scn.iter_stream():
        if kind == "gps":
            assert any(np.allclose(payload[0], p) for p in scn.attack.pos)
        else:
            accel, gyro, heading = payload
            assert np.shape(accel) == (2,)
            assert np.isscalar(gyro) or np.ndim(gyro) == 0
            assert np.isscalar(heading) or np.ndim(heading) == 0


def test_clean_flight_does_not_alarm():
    scn = scenario_mod.build(**scenario_mod.PRESETS["clean"], seed=0)
    log = run_detector(scn)
    assert log["flag_time"] is None, "false alarm on clean flight"


def test_jump_is_caught_within_one_epoch():
    scn, log, score = trial("jump", seed=0)
    assert score.detected
    assert score.latency_s is not None and score.latency_s <= scn.dt_gps


def test_slow_drift_is_not_caught_and_that_is_documented():
    """Phase 0 genuinely cannot catch a 0.5 m/s walk-off. Pinning the known
    limitation means a future change that fixes it will trip this test loudly
    rather than passing unnoticed."""
    _scn, _log, score = trial("drift", seed=0)
    assert not score.detected, "drift now detected -- update README and Phase 1"


def test_step_facade_matches_the_split_api():
    """The spec's single `step(gps_fix, imu_sample)` must agree exactly with
    the tight-loop `step_imu`/`step_gps` path."""
    scn = scenario_mod.build(spoof="jump", seed=1, duration=60.0)
    dt = 1.0 / scn.fs_imu

    split = SpoofDetector(fs_imu=scn.fs_imu, dt_gps=scn.dt_gps)
    facade = SpoofDetector(fs_imu=scn.fs_imu, dt_gps=scn.dt_gps)
    split_out, facade_out = [], []

    for kind, _t, *payload in scn.iter_stream():
        if kind == "imu":
            accel, gyro, heading = payload
            split.step_imu(accel, gyro, heading, dt)
            facade.step(imu_sample={"accel": accel, "gyro": gyro,
                                    "heading": heading, "dt": dt})
        else:
            split_out.append(split.step_gps(payload[0], scn.dt_gps)["score"])
            facade_out.append(
                facade.step(gps_fix={"pos": payload[0], "dt": scn.dt_gps})["score"])

    assert np.allclose(split_out, facade_out)


def test_report_contract():
    """Whatever else changes, the result dict keeps the documented keys."""
    _scn, log, _score = trial("jump", seed=0)
    det = log["detector"]
    out = det.step()
    for key in ("spoofed", "score", "reason"):
        assert key in out
    assert isinstance(out["spoofed"], bool)
    assert isinstance(out["reason"], str)


def test_imu_noise_scales_with_sample_rate():
    """Per-sample sigma should follow density * sqrt(fs): a faster IMU is
    noisier per sample but no worse once integrated."""
    # Due north, so the body frame coincides with the navigation frame and the
    # IMU output can be differenced against truth directly.
    traj = trajectory.generate("line", duration=40.0, fs=100.0, bearing_deg=0.0)
    assert np.allclose(traj.heading, 0.0, atol=1e-9)

    quiet = sensors.ImuSpec(accel_noise_density=0.0, accel_bias_sigma=0.0,
                            accel_bias_rw=0.0, gyro_noise_density=0.0,
                            gyro_bias_sigma=0.0, gyro_bias_rw=0.0)
    imu = sensors.simulate_imu(traj, quiet, np.random.default_rng(0))
    assert np.allclose(imu.accel, traj.acc, atol=1e-9)

    loud = sensors.ImuSpec(accel_noise_density=1e-2, accel_bias_sigma=0.0,
                           accel_bias_rw=0.0, gyro_noise_density=0.0,
                           gyro_bias_sigma=0.0, gyro_bias_rw=0.0)
    imu = sensors.simulate_imu(traj, loud, np.random.default_rng(0))
    residual = np.linalg.norm(imu.accel - traj.acc, axis=1)
    # Per-axis sigma is density * sqrt(fs); the norm of two such axes is
    # Rayleigh, whose mean is sigma * sqrt(pi/2), not sigma * sqrt(2).
    expected = 1e-2 * np.sqrt(100.0) * np.sqrt(np.pi / 2.0)
    assert 0.9 * expected < residual.mean() < 1.1 * expected


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("  PASS  {}".format(fn.__name__))
        except Exception:
            failed += 1
            print("  FAIL  {}".format(fn.__name__))
            traceback.print_exc()
    print("\n{}/{} passed".format(len(tests) - failed, len(tests)))
    raise SystemExit(1 if failed else 0)
