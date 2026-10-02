# GNSS Spoofing Detection Layer

Detecting GPS spoofing on a UAV by cross-checking the GNSS fix against
inertial dead-reckoning.

**Status: Phase 0 complete.** Jump and replay attacks are caught reliably with
no false alarms on clean flight. Slow walk-off attacks are *not* caught below
about 2.5 m/s, and the measured boundary is reported below rather than hidden
— locating it is what the next two phases are for.

No RF is transmitted anywhere in this project. All spoofing is injected in
software. See [Legality](#legality).

---

## Why spoofing and not jamming

Jamming denies the signal, and a drone notices immediately: the fix drops out
and the autopilot falls back on inertial navigation. Spoofing feeds a *false
but plausible* position that the autopilot keeps trusting, so the aircraft
flies confidently to the wrong place. It is the subtler problem, the more
dangerous one, and far less solved — which makes it a live counter-UAS / EW
concern for anyone flying in contested airspace.

## The idea in one table

The whole detector rests on GPS and an IMU failing in *opposite* ways:

|                       | GPS                                     | IMU                                            |
| --------------------- | --------------------------------------- | ---------------------------------------------- |
| Error over time       | Bounded, a few metres, forever          | Grows as t^1.5 from noise, t^2 from bias       |
| Forgeable?            | **Yes** — a ~20 W signal from 20 000 km | **No** — it measures the airframe's own physics |

The IMU is an *unforgeable but short-sighted* witness. Over seconds to tens of
seconds it is accurate enough to audit GPS before its own drift swamps the
comparison. The detector lives inside that window, continuously asking: *GPS
claims I moved 12 m north; my accelerometers felt a motion consistent with 3 m
north — and only one of you can be lied to.*

## Architecture

```
                 body-frame accel, gyro          AHRS heading
 IMU (100 Hz) ─────────────────────────┐        (magnetometer-based,
                                       │         INDEPENDENT of GNSS)
                                       v              │
                            ┌──────────────────────┐  │
                            │ strapdown            │<─┘
                            │ dead-reckoning       │
                            │ (trapezoidal)        │
                            └──────────┬───────────┘
                                       │ independent position estimate
                                       v
 GNSS (1 Hz) ──────────────────> ( compare ) ──> residual
 position fix                          │
   ▲                                   v
   │                        ┌────────────────────────┐
   │                        │ normalise by LEARNED   │  <- adaptive threshold
   │                        │ median + robust scale  │
   │                        └──────────┬─────────────┘
   │                                   v
   │                        ┌────────────────────────┐
   │                        │ CUSUM change detector  │
   │                        └──────────┬─────────────┘
   │                                   v
   │                           {spoofed, score, reason}
   │                                   │
   └───── anchor (alpha-beta) <─────────┘  cut on alarm, then coast inertial
```

The detector only ever sees what a real flight computer would: body-frame IMU,
an AHRS heading, and a GPS position. It never sees ground truth. Keeping that
boundary honest is what makes the Phase 3 port to a live serial feed a drop-in
replacement of one loop — and what prevents accidentally writing a detector
that cheats.

## Three design decisions worth defending

**1. Alarm on a *change* in the residual, not on a large residual.** The
residual is never zero: accelerometer turn-on bias puts dead-reckoning at a
standing lag behind GPS forever, and attitude error adds more during
manoeuvres. Threshold the raw residual and you must set the threshold above
all of that, throwing away your sensitivity. Instead the detector learns the
residual's own median and robust scale during quiet flight and alarms on
departures from it. A standing nuisance offset is absorbed; a new one is not.

**2. Heading comes from the AHRS, never from GPS course over ground.** An
early version derived heading from successive GPS fixes. That is wrong on two
counts: it is barely observable (3.5 m of position noise over a 12 m step is
17° of course error), and more importantly it hands the attacker the very
reference being used to audit them. A magnetometer-based heading is outside
the spoofer's reach, and that independence is the entire source of the
detector's leverage.

**3. The velocity-anchoring gain is the crux of the whole problem.** A
walk-off spoof is a position ramp, and any filter that freely adapts its
velocity will adopt the attacker's ramp as truth — after which the residual
collapses back into the noise and the attack is invisible. But the
accelerometers *know* the aircraft never accelerated to that new velocity.
Velocity is precisely the state to trust the IMU on and GPS least. Phase 2
stops setting this gain by hand: it falls out of the Kalman covariance.

## Results

16 independent seeds per configuration, 240 s survey (lawnmower) flight at
12 m/s, 100 Hz IMU, 1 Hz GPS at 2.5 m per-axis noise. Detection rate is per
*run*; a run only counts as detected if it had not already false-alarmed.
False alarms are counted as **alarm onsets per hour** of clean flight, not as
flagged epochs — the detector latches, so counting epochs would turn one false
alarm into an 80% "rate" and mean nothing.

| spoof  | detection | FA / hour | latency p50 | p90   | max   |
| ------ | --------- | --------- | ----------- | ----- | ----- |
| clean  | —         | **0.00**  | —           | —     | —     |
| jump   | **100 %** | 0.00      | 0 s         | 1 s   | 1 s   |
| replay | **100 %** | 0.00      | 3 s         | 3 s   | 3 s   |
| drift  | **0 %**   | 0.00      | —           | —     | —     |

Jump (50 m) is caught on the first spoofed fix; with a 1 Hz receiver, 0–1 s is
the floor. Replay (25 s lag) takes 3 s, because the handover is continuous by
construction and the real and replayed tracks need a moment to diverge. Zero
false-alarm onsets across 56 min of clean flight.

### Where Phase 0 gives out — the honest headline

Walk-off detection by rate (12 seeds each):

| walk-off rate  | ≤ 1.0 m/s | 1.5 m/s | 2.5 m/s | 4.0 m/s |
| -------------- | --------- | ------- | ------- | ------- |
| detection      | **0 %**   | 25 %    | 50 %    | 100 %   |
| median latency | —         | 12 s    | 9 s     | 8 s     |
| offset when caught | —     | 18 m    | 23 m    | 32 m    |

A patient attacker walking the fix off at 1 m/s is never caught. The mechanism
is visible in `results/scenario_drift.png`: the residual spikes to 7.6 m about
ten seconds after onset, then **decays back into the noise** while the true
induced error keeps growing past 77 m. The filter's velocity state absorbed the
ramp.

The deeper reason is the argument for Phase 2: a position-domain residual
normalised by its own *observed* scatter cannot detect a slow ramp, because a
slow ramp is indistinguishable from the very nuisance drift the adaptive scale
exists to absorb. Escaping it needs a threshold derived from a *model* of
sensor noise rather than from observed scatter — which is exactly what an EKF's
innovation covariance provides.

### CUSUM vs the obvious baseline

The spec's rule — flag after N consecutive fixes above M sigma — is
implemented as `--mode nsigma` so the comparison is on the record. On jump they
tie at 100%. On a 0.5 m/s walk-off, accumulating evidence wins outright:

| method               | best drift detection | at FA/hour |
| -------------------- | -------------------- | ---------- |
| n-sigma (3 × 2.0σ)   | 37 %                 | 6.5        |
| CUSUM                | **75 %**             | 10.8       |

A single epoch of a slow walk-off never looks suspicious, so a per-epoch
threshold has nothing to fire on; CUSUM integrates the small persistent bias
instead. (ROC caveat: at 8 trials the resolution floor is one onset in
1680 clean epochs, i.e. ~2 FA/hour, so a plotted `0.00` means "below 2", not
zero. The CLI prints this floor.)

### Choosing the gains by measurement

`vel_gain` is the crux parameter, and measuring it over too narrow an envelope
would have picked the wrong value. On 240 s flights, `vel_gain = 0` looks
strictly best:

| vel_gain | FA/hour (240 s) | drift 0.5 | drift 1.0 | drift 2.5 |
| -------- | --------------- | --------- | --------- | --------- |
| 0.000    | 0.00            | 0 %       | **17 %**  | **58 %**  |
| 0.002    | 0.00            | 0 %       | 8 %       | 58 %      |
| **0.005**| 0.00            | 0 %       | 0 %       | 50 %      |
| 0.020    | 0.00            | 0 %       | 0 %       | 33 %      |
| 0.050    | 0.00            | 0 %       | 0 %       | 0 %       |

But extend the flight and `vel_gain = 0` falls apart — with nothing correcting
velocity, error from accelerometer bias grows without bound:

| FA/hour on clean flight | 240 s | 480 s | 900 s |
| ----------------------- | ----- | ----- | ----- |
| vel_gain = 0.000        | 0.00  | 3.21  | 1.66  |
| vel_gain = 0.002        | 0.00  | 2.41  | 0.83  |
| **vel_gain = 0.005**    | 0.00  | 0.00  | 0.00  |
| vel_gain = 0.020        | 0.00  | 0.00  | 0.00  |

0.005 is the smallest gain that stays quiet out to 900 s, so that is the
default — the best-performing value on the short flight would have been a bug
in waiting. Validate over the operational envelope, not one scenario. The
threshold `cusum_h = 16` was set the same way: the peak clean-flight
accumulator over 30 runs was median 5.9, p90 9.3, max 10.5, so 16 leaves ~1.5x
margin on the worst observed clean flight. Both are properties of *this*
vehicle's noise, not universal constants — re-measure them when the airframe,
IMU or flight profile changes.

Figures in `results/`: `scenario_{clean,jump,drift,replay}.png`, `roc.png`,
`latency.png`, `drift_sweep.png`.

## Two bugs that shaped the design

Both are kept here because they were found by measurement, not by reading the
code, and both are the kind of thing that silently ruins this class of
detector.

**Forward-Euler integration cost 17 m.** Free-running the dead-reckoning with
a *noise-free* IMU should give near-zero error. It gave 17 m at 120 s.
Forward Euler leaves a half-sample lag in heading, a lagging heading
misrotates the accelerometers, and the misrotation integrates into position
error the same size as the attacks being detected. Trapezoidal integration
brings it to 2.5 cm/s of drift — about 2.5 m at 120 s, below the GPS noise
floor — for two extra adds per sample.

**Graded trust created a positive feedback loop.** Scaling the anchoring gain
down as suspicion rose looked prudent: don't let a spoofer drag the reference
it is being measured against. But weaker anchoring raises the residual, a
higher residual raises suspicion, and that weakens anchoring further. Cost:
10 false alarms in 12 clean runs while the underlying residual was perfectly
healthy (z median 1.19, p99 3.13). The same trap appeared a second time via
the statistics-learning gate, which froze the scale estimate when the
accumulator rose, biasing z upward. Both now use non-accumulating tests, and
the attacker's ability to drag the reference is bounded structurally instead —
by how small `vel_gain` is.

## Run

```bash
python -m eval.run --scenario jump            # the Phase 0 deliverable
python -m eval.run --scenario all --trials 16 # full results table
python -m eval.run --drift-sweep              # where the detector gives out
python -m eval.run --gain-sweep               # the vel_gain trade-off, measured
python -m eval.run --roc --mode both          # ROC: CUSUM vs n-sigma baseline
python -m eval.run --scenario drift --path orbit --mode nsigma
```

Requires only numpy, scipy and matplotlib (`pip install -r requirements.txt`).

```bash
python -m tests.test_smoke                    # 9 tests, no pytest needed
```

The tests pin the properties that break silently: seed reproducibility, that
the sensor stream never leaks ground truth, that the spec's `step()` facade
agrees exactly with the tight-loop API, and — deliberately — that a 0.5 m/s
walk-off is *not* detected, so whichever phase fixes that trips the test loudly
instead of passing unnoticed.

## Layout

```
sim/        trajectory.py  ground-truth flight paths (survey/orbit/line/hover)
            sensors.py     IMU, AHRS and GPS error models
            spoofer.py     the adversary: jump / walk-off drift / replay
            scenario.py    assembles one replayable run; owns the truth boundary
detector/   consistency.py SpoofDetector: step_imu() / step_gps() / step()
eval/       metrics.py     detection rate, false-alarm rate, latency
            plots.py       figures
            run.py         CLI; the only place sim and detector touch
```

## Limitations and future work

Stated plainly, because overclaiming here is worse than the gaps themselves.

- **Software-injected spoofing is not RF spoofing.** This detector works at
  the position-solution layer only. A real attack also leaves signal-layer
  fingerprints — C/N0 and AGC anomalies, carrier-phase discontinuity, loss of
  multipath diversity, inconsistent satellite geometry — none of which are
  modelled. A u-blox M8N's own `spoofDetState` flag watches some of them, so
  this approach is *complementary* to receiver-level checks, not a
  replacement.
- **Slow walk-off is unsolved below ~2.5 m/s.** Quantified above. This is the
  threat that matters most, and Phase 2 is aimed squarely at it.
- **IMU noise parameters are datasheet-order, not measured.** They are
  therefore assumptions, and tuning a detector against assumed noise risks a
  self-fulfilling result. Next step is an Allan-variance run on the real
  MPU-6050 to extract actual bias instability and random-walk coefficients and
  feed those back in.
- **2-D horizontal, flat-earth, gravity-free.** Altitude is held, and a level
  2-D model puts gravity entirely on body-z. In 3-D, attitude error leaking
  gravity into the horizontal axes becomes the dominant dead-reckoning error:
  1° of tilt injects 0.17 m/s², four times the accelerometer noise modelled
  here. Phase 1's job.
- **GPS error is white.** Real receiver error is correlated over seconds to
  minutes (ionosphere, multipath), which makes a slow spoof harder to separate
  from a bad sky view.
- **Synthetic trajectories.** Real vehicle dynamics should come from public
  PX4/ArduPilot flight logs.
- **A slow attack can partly teach the detector to accept it**, since
  residuals below the outlier-rejection threshold still enter the learned
  statistics. Bounded by the 40-fix window and by the latch, but real.

### Roadmap

- **Phase 1** — full 3-D strapdown with quaternion attitude and gravity
  compensation; IMU noise measured by Allan variance on the real part; real
  flight-log trajectories.
- **Phase 2** — EKF fusing IMU and GPS, with **innovation monitoring**. In a
  correctly tuned filter the innovation is zero-mean with known covariance
  `S = HPHᵀ + R`, so the normalised innovation squared follows a chi-squared
  distribution and the threshold comes from statistics rather than from
  observed scatter. A spoofer injects a persistent bias, which a chi-squared
  test on accumulated innovation detects at far smaller amplitudes than Phase
  0 can reach. The literature calls this INS-aided RAIM.
- **Phase 3** — port to ESP32 reading a real MPU-6050 over I²C and a u-blox
  NEO-M8N over UART (parsing UBX binary, not NMEA, to reach `UBX-NAV-STATUS`
  and per-satellite C/N0); spoof injected into the incoming stream in
  software, on the wire, never over the air.
- **Phase 4** — TEXBAT benchmark; VIO cross-check; swarm cross-validation;
  autonomous fail-safe response (drop GPS, navigate inertially, return home).

## Legality

**Never transmit on GNSS frequencies.** Broadcasting fake GPS signals is
illegal in essentially every jurisdiction and interferes with real navigation,
including aviation and emergency services. Every attack in this repository is
applied to a position solution in software. That is not only legal, it is
better for the engineering: holding ground truth is the only way to measure
detection latency to the sample, which an over-the-air test could never give
you.
