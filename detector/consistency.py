"""Phase 0 detector: anchored dead-reckoning + residual change detection.

The idea in one sentence
-----------------------
Keep an independent position estimate driven only by the IMU, lightly
anchored to GPS while GPS looks honest, and watch the residual between the
two for a *change* rather than for a large value.

What the detector is given, and why
-----------------------------------
Body-frame acceleration, turn rate, and a heading from the autopilot's
attitude filter. It is NOT given ground truth, and it does not derive
heading from GPS course over ground -- that would hand the attacker the
reference being used to audit it. The AHRS heading comes from a
magnetometer and is therefore outside the spoofer's reach, which is the
whole reason the cross-check has any teeth.

Why alarm on a change and not on a large value
----------------------------------------------
The residual is never zero. The accelerometer has a turn-on bias, so
dead-reckoning sits at a small steady-state lag behind GPS forever, and
attitude error adds more during manoeuvres. Threshold the raw residual and
you must set the threshold above all of that, throwing away your
sensitivity. Instead we learn the residual's own median and scale while
things are quiet, then alarm when it departs from what this airframe's own
sensors normally disagree by. That is what an "adaptive threshold" buys: a
standing nuisance offset is absorbed, a new one is not.

The gain that decides who wins
------------------------------
Two gains anchor the estimate to GPS: `pos_gain` on position and `vel_gain`
on velocity. `vel_gain` is deliberately tiny. A walk-off spoof is a
position ramp, and a filter that freely adapts its velocity simply adopts
the attacker's ramp as truth, after which the residual collapses back into
the noise. But the accelerometers *know* the aircraft never accelerated to
that new velocity -- velocity is precisely the state to trust the IMU on and
GPS least. Raising `vel_gain` makes clean flight quieter and slow spoofs
invisible; lowering it does the opposite. That trade-off is the core of the
problem, and the reason Phase 2 replaces these hand-set gains with a Kalman
gain derived from an actual covariance.

Trust is graded, not binary: as suspicion rises the anchoring gain is scaled
down, so a spoofer cannot drag the reference it is being measured against.
Once latched, anchoring stops entirely and the estimate coasts on inertial
alone -- which is also what a downstream fail-safe wants.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

#: MAD -> sigma conversion for Gaussian data.
_MAD_TO_SIGMA = 1.4826


class SpoofDetector:
    """Streaming GNSS spoof detector.

    Call `step_imu()` at the IMU rate and `step_gps()` on each fix, or use
    the `step()` facade.

    Parameters
    ----------
    mode : {"cusum", "nsigma"}
        "nsigma" is the obvious baseline: flag after `n_consecutive` fixes
        above `n_sigma`. "cusum" accumulates evidence instead, so it can
        catch a bias far smaller than its own noise given enough time. Both
        are kept so the evaluation can show the difference.
    cusum_k : slack, in sigmas. Residual below this is treated as free. Set
        it just above the clean-flight mean of the normalised residual
        (about 1.25 for a 2-D Rayleigh) or the accumulator ratchets up on
        noise alone.
    cusum_h : alarm threshold on the accumulator. The default is measured,
        not guessed: over 30 clean 240 s flights the peak accumulator value
        was median 5.9, p90 9.3, max 10.5, so 16 leaves about 1.5x margin on
        the worst observed clean run. Re-measure it if the airframe, the IMU
        or the flight profile changes -- it is a property of the vehicle's own
        noise, not a universal constant. The ROC sweeps it.
    pos_gain, vel_gain : anchoring gains, alpha-beta form. See module docs.
        `vel_gain` is measured, not guessed, and the measurement is
        instructive. On 240 s flights vel_gain=0 scored best (60% detection on
        a 2.5 m/s walk-off vs 30% at 0.02) with no false alarms, so a
        240 s-only study would have chosen it. Extend the flight and it falls
        apart: vel_gain=0 raises 3.2 false alarms/hour over 480 s flights and
        1.7/hour over 900 s, because velocity error from accelerometer bias
        grows without bound when nothing corrects it. 0.005 is the smallest
        gain with zero false alarms out to 900 s, and still catches a 2.5 m/s
        walk-off half the time. Validate over the operational envelope, not
        one flight length.
    align_pos_gain, align_vel_gain : the same gains during warm-up. Velocity
        has to be seeded from GPS, which is noisy, so the filter needs to
        converge fast before it can afford to be sensitive. Real inertial
        systems do the same: coarse alignment, then tighten.
    """

    def __init__(self, fs_imu: float = 100.0, dt_gps: float = 1.0,
                 gps_sigma_m: float = 2.5,
                 mode: str = "cusum",
                 cusum_k: float = 1.5, cusum_h: float = 16.0,
                 n_sigma: float = 5.0, n_consecutive: int = 3,
                 pos_gain: float = 0.30, vel_gain: float = 0.005,
                 align_pos_gain: float = 0.60, align_vel_gain: float = 0.25,
                 scale_window: int = 40, warmup_fixes: int = 30,
                 learn_outlier_z: float = 4.0,
                 init_fixes: int = 6, settle_fixes: int = 10,
                 latch: bool = True):
        if mode not in ("cusum", "nsigma"):
            raise ValueError("mode must be 'cusum' or 'nsigma', got " + repr(mode))

        self.fs_imu = float(fs_imu)
        self.dt_gps = float(dt_gps)
        self.gps_sigma_m = float(gps_sigma_m)
        self.mode = mode
        self.cusum_k = float(cusum_k)
        self.cusum_h = float(cusum_h)
        self.n_sigma = float(n_sigma)
        self.n_consecutive = int(n_consecutive)
        self.pos_gain = float(pos_gain)
        self.vel_gain = float(vel_gain)
        self.align_pos_gain = float(align_pos_gain)
        self.align_vel_gain = float(align_vel_gain)
        self.learn_outlier_z = float(learn_outlier_z)
        self.scale_window = int(scale_window)
        self.warmup_fixes = int(warmup_fixes)
        self.init_fixes = int(init_fixes)
        self.settle_fixes = int(settle_fixes)
        self.latch = bool(latch)
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        self.t = 0.0
        self.psi = 0.0                   # heading in use [rad], from the AHRS
        self.pos = np.zeros(2)           # dead-reckoned position [m]
        self.vel = np.zeros(2)           # dead-reckoned velocity [m/s]
        self.residual = np.zeros(2)
        self.z = 0.0                     # normalised residual
        self.cusum = 0.0
        self.sigma = self.gps_sigma_m
        self.run_len = 0
        self.n_fixes = 0
        self.flagged = False
        self.flag_time = None
        self.turn_rate = 0.0             # low-passed, diagnostic
        self._initialised = False
        self._init_fits: list = []       # (t, pos) pairs used to seed velocity
        self._accel_prev = None          # for trapezoidal integration
        self._buf: deque = deque(maxlen=self.scale_window)

    @property
    def warming(self) -> bool:
        """True while aligning. The detector is not yet on duty: it anchors
        hard, learns nothing, and cannot raise an alarm."""
        return self.n_fixes <= self.warmup_fixes

    @property
    def threshold(self) -> float:
        return self.cusum_h if self.mode == "cusum" else self.n_sigma

    @property
    def _gains(self) -> tuple:
        if self.warming:
            return self.align_pos_gain, self.align_vel_gain
        return self.pos_gain, self.vel_gain

    # -- inertial propagation ---------------------------------------------

    def step_imu(self, accel_body, gyro_z, heading, dt=None) -> None:
        """Strapdown update: rotate the accelerometers into the navigation
        frame using the AHRS heading, then integrate twice.

        Integration is trapezoidal, not forward Euler. That is not
        fastidiousness: forward Euler on this signal leaves a systematic
        velocity error that integrates to ~17 m over two minutes even with a
        noise-free IMU -- the same size as the attacks being detected.
        Trapezoidal brings it to a few centimetres for two extra adds per
        sample, so there is no reason to run anything cruder on real
        hardware either.

        `gyro_z` is low-passed into `turn_rate` for diagnostics; Phase 2
        promotes it to a propagated state with its bias estimated.
        """
        dt = 1.0 / self.fs_imu if dt is None else float(dt)
        self.t += dt
        self.psi = float(heading)
        self.turn_rate += 0.02 * (float(gyro_z) - self.turn_rate)

        c, s = math.cos(self.psi), math.sin(self.psi)
        ax, ay = float(accel_body[0]), float(accel_body[1])
        a_nav = np.array([c * ax - s * ay, s * ax + c * ay])
        a_mid = a_nav if self._accel_prev is None else 0.5 * (self._accel_prev + a_nav)
        self._accel_prev = a_nav

        if not self._initialised:
            return  # no position anchor yet, nothing to propagate from

        self.pos += self.vel * dt + 0.5 * a_mid * dt * dt
        self.vel += a_mid * dt

    # -- GPS epoch: compare, decide, then anchor ---------------------------

    def step_gps(self, gps_pos, dt_gps=None) -> dict:
        gps_pos = np.asarray(gps_pos, dtype=float)
        dt_gps = self.dt_gps if dt_gps is None else float(dt_gps)
        self.n_fixes += 1

        # Dead-reckoning cannot start until velocity is known, and an IMU
        # cannot observe its own velocity -- only changes to it. So velocity
        # must be seeded from GPS. Differencing two adjacent fixes gives
        # sigma_v = sigma_p * sqrt(2) / dt = 3.5 m/s, which takes longer to
        # settle than the whole warm-up; a least-squares fit over
        # `init_fixes` epochs cuts that by roughly a factor of four.
        if not self._initialised:
            self._init_fits.append((self.t, gps_pos.copy()))
            if len(self._init_fits) < max(self.init_fixes, 2):
                return self._report("init: collecting fixes {}/{}".format(
                    len(self._init_fits), self.init_fixes))
            self.pos, self.vel = self._seed_state()
            self._initialised = True
            return self._report("init: state seeded by least squares")

        resid = gps_pos - self.pos
        self.residual = resid

        # Normalise against what this airframe's residual normally looks
        # like. Centring on the learned median absorbs the standing lag from
        # accelerometer bias; the floor stops sigma collapsing below the
        # receiver's own noise and manufacturing significance.
        mu, scale = self._learned_stats()
        self.sigma = max(scale, self.gps_sigma_m)
        self.z = float(np.linalg.norm(resid - mu) / self.sigma)

        if self.mode == "cusum":
            self.cusum = max(0.0, self.cusum + (self.z - self.cusum_k))
            score, trip = self.cusum, self.cusum > self.cusum_h
        else:
            self.run_len = self.run_len + 1 if self.z > self.n_sigma else 0
            score, trip = self.z, self.run_len >= self.n_consecutive

        # Evidence gathered while aligning is not evidence of anything -- the
        # alignment transient dwarfs it. Hold the accumulator down until the
        # detector is on duty, or it trips the instant warm-up ends.
        if self.warming:
            self.cusum = 0.0
            self.run_len = 0
            score, trip = 0.0, False

        if trip:
            if not self.flagged:
                self.flag_time = self.t
            self.flagged = True
        alarmed = self.flagged if self.latch else trip

        # Anchoring is all-or-nothing, and that is a deliberate correction of
        # an earlier design. Scaling the anchoring gain down as suspicion
        # rose looked prudent -- don't let a spoofer drag the reference it is
        # being measured against -- but it closes a positive feedback loop:
        # weaker anchoring raises the residual, a higher residual raises
        # suspicion, which weakens anchoring further. Measured cost was 10
        # false alarms in 12 clean runs, despite the underlying residual
        # being perfectly healthy (z median 1.19, p99 3.1). The attacker's
        # ability to drag the reference is instead bounded structurally, by
        # how small `vel_gain` is.
        if not alarmed:
            self._anchor(resid, dt_gps, 1.0)

        # Learn the quiet-flight statistics, skipping the alignment transient
        # and rejecting individual outliers. The rejection test is on the
        # INSTANTANEOUS z, deliberately not on the accumulated score: gating
        # it on the accumulator was a second instance of the same feedback
        # trap, because a rising accumulator froze the scale estimate, which
        # biased z upward, which raised the accumulator further. A per-sample
        # outlier test cannot lock itself out.
        settled = self.n_fixes > self.init_fixes + self.settle_fixes
        if settled and not alarmed and self.z < self.learn_outlier_z:
            self._buf.append(resid.copy())

        return self._report(self._reason(alarmed, score))

    def _anchor(self, resid, dt_gps, trust) -> None:
        """Alpha-beta pull toward GPS, scaled by how much we trust it."""
        pos_gain, vel_gain = self._gains
        self.pos = self.pos + pos_gain * trust * resid
        self.vel = self.vel + (vel_gain * trust / dt_gps) * resid

    # -- helpers -----------------------------------------------------------

    def _seed_state(self) -> tuple:
        """Least-squares constant-velocity fit through the first few fixes,
        evaluated at the most recent one."""
        ts = np.array([t for t, _ in self._init_fits])
        ps = np.array([p for _, p in self._init_fits])
        design = np.stack([np.ones_like(ts), ts - ts[-1]], axis=1)
        coef, *_ = np.linalg.lstsq(design, ps, rcond=None)
        return coef[0].copy(), coef[1].copy()      # position, velocity

    def _learned_stats(self) -> tuple:
        """Median offset and robust scale of the recent quiet residual."""
        if len(self._buf) < 8:
            return np.zeros(2), self.gps_sigma_m
        buf = np.asarray(self._buf)
        mu = np.median(buf, axis=0)
        scale = _MAD_TO_SIGMA * float(np.median(np.abs(buf - mu)))
        return mu, scale

    def _reason(self, alarmed, score) -> str:
        if self.warming:
            return "warm-up {}/{}".format(self.n_fixes, self.warmup_fixes)
        just_tripped = (alarmed and self.flag_time is not None
                        and abs(self.t - self.flag_time) < 1e-9)
        if just_tripped:
            if self.z > 4.0 * self.cusum_k:
                return "step change in residual: {:.1f} sigma".format(self.z)
            return "sustained residual: evidence {:.1f} > {:.1f}".format(
                score, self.threshold)
        if alarmed:
            return "latched since t={:.1f}s".format(self.flag_time)
        if score > 0.3 * self.threshold:
            return "residual elevated: evidence {:.1f}/{:.1f}".format(
                score, self.threshold)
        return "consistent"

    def _report(self, reason) -> dict:
        if self.latch:
            spoofed = self.flagged
        elif self.mode == "cusum":
            spoofed = self.cusum > self.cusum_h
        else:
            spoofed = self.run_len >= self.n_consecutive
        return {
            "spoofed": bool(spoofed and not self.warming),
            "score": float(self.cusum if self.mode == "cusum" else self.z),
            "reason": reason,
            "t": float(self.t),
            "z": float(self.z),
            "sigma_m": float(self.sigma),
            "residual_m": float(np.linalg.norm(self.residual)),
            "dr_pos": self.pos.copy(),
        }

    # -- plan-facing facade ------------------------------------------------

    def step(self, gps_fix=None, imu_sample=None) -> dict:
        """Single entry point matching the project spec.

        imu_sample : {"accel": (2,), "gyro": float, "heading": float,
                      "dt": float (optional)}
        gps_fix    : {"pos": (2,), "dt": float (optional)} or a bare (2,)

        Returns {"spoofed", "score", "reason", ...}. For tight loops call
        `step_imu` / `step_gps` directly and skip the dict packing.
        """
        if imu_sample is not None:
            self.step_imu(imu_sample["accel"], imu_sample["gyro"],
                          imu_sample["heading"], imu_sample.get("dt"))
        if gps_fix is not None:
            if isinstance(gps_fix, dict):
                return self.step_gps(gps_fix["pos"], gps_fix.get("dt"))
            return self.step_gps(gps_fix)
        return self._report("imu only")
