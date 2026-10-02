# GNSS Spoofing Detection Layer — Build Plan & Starter Prompt

**One-line objective:** Build a proof-of-concept module that detects (and rejects) GPS/GNSS *spoofing* by cross-checking the GNSS fix against inertial dead-reckoning and sensor fusion — because jamming is denial, but spoofing is deception, and it's the harder, less-solved problem for defense UAVs.

> **How to use this file:**
> 1. As **your roadmap** — follow the phases top to bottom.
> 2. As a **prompt for a coding assistant** — paste Section A (the prompt) to kick off the scaffold, then feed it one phase at a time from Section C.

---

## ⚠️ Section 0 — Legal / safety (read first)
**Never transmit on GNSS frequencies.** Broadcasting fake GPS signals is illegal almost everywhere and interferes with real navigation. You will **simulate/inject** spoofing entirely in software and over wires — this is legal, reproducible, and actually *better* for measuring performance. No RF transmission, ever.

---

## Section A — Starter prompt (paste this to a coding agent)

> I'm an ECE student building a **GNSS spoofing-detection layer** for drones as a portfolio project. Spoofing feeds a drone a *false but plausible* GPS position; my detector catches it by cross-checking GPS against inertial dead-reckoning and (later) an EKF.
>
> Scaffold a clean Python repo named `gnss-spoof-detect` with:
> - `sim/` — a spoofing-injection harness: load a ground-truth trajectory (real recorded GPS log *or* a synthetic path), then generate a "spoofed" GPS stream by applying one of: (a) a sudden position **jump**, (b) a slow position **walk-off drift**, (c) a **replay** offset. Output: time-series of {true_pos, gps_pos, imu_accel, imu_gyro}.
> - `detector/` — a `SpoofDetector` class with a `step(gps_fix, imu_sample)` method returning `{spoofed: bool, score: float, reason: str}`. Start with a **consistency check**: integrate IMU to get expected displacement, compare to GPS displacement; flag when divergence exceeds an adaptive threshold for N consecutive samples.
> - `eval/` — compute detection rate, false-alarm rate, and detection latency; plot an ROC curve and a latency chart with matplotlib.
> - `README.md` — problem statement, architecture diagram, how to run, results.
>
> Use only numpy/scipy/matplotlib (add filterpy later for the EKF). Keep it modular so I can later swap the simulated IMU/GPS for a live serial feed from an ESP32. Write Phase 0 first: the sim harness + the consistency-check detector catching a **jump** spoof, with a plot.

---

## Section B — Hardware (you can start with ZERO hardware)

**Phase 0–2 need no hardware** — pure software with logged/synthetic data. Start today.

For the live bench demo (Phase 3), using parts you likely have + one cheap buy:
| Part | Purpose | Notes |
|---|---|---|
| ESP32 / Teensy 4.1 | Run detector on-device | You own these |
| MPU-6050 / 9250 IMU | Inertial reference | You own this |
| **u-blox NEO-M8N / M9N GPS** | Real GNSS fixes + built-in spoof flags | ~₹800–1500; the one thing to buy |
| LED / buzzer | Spoof alarm output | Trivial |

The u-blox modules expose `UBX-NAV-STATUS` → `spoofDetState` and AGC/C-N0 data — you can *read and extend* their own indicators (great talking point).

---

## Section C — Phased milestones

### Phase 0 — Sim harness + v0 detector (2–4 days) ← start here
- Build the spoofing-injection simulator (jump / drift / replay).
- Implement the IMU-vs-GPS **consistency check** detector.
- **Deliverable:** detector catches a *jump* spoof; one plot. Repo is live on GitHub.
- ✅ After this you can **honestly say "currently building."**

### Phase 1 — Dead-reckoning + the hard case (3–5 days)
- Proper IMU dead-reckoning (orientation + double-integrate accel, with drift modeling).
- Tune to catch the **slow walk-off drift** (the hard case; jumps are easy).
- **Deliverable:** detection-rate + false-alarm numbers for jump vs drift.

### Phase 2 — EKF + innovation monitoring (4–7 days)
- Fuse IMU+GPS in a simple EKF (filterpy); monitor the **innovation** (measurement residual).
- Sustained large innovation = GPS is lying → the rigorous, citable method.
- **Deliverable:** ROC curve, detection latency chart, adaptive thresholds.

### Phase 3 — Live hardware bench demo (4–7 days)
- Port detector to ESP32/Teensy reading real IMU + u-blox GPS over serial.
- Inject spoof into the *incoming* GPS stream (software, on the wire — no RF).
- **Deliverable:** 60–90 s demo video of the LED/buzzer firing on a live injected spoof.

### Phase 4 — Stretch (pick 1–2, signals depth)
- **TEXBAT benchmark** (Texas Spoofing Test Battery) — "validated on the standard academic dataset."
- **VIO/SLAM cross-check** — reuse your SLAM work: vision vs GPS disagreement = spoof.
- **Swarm cross-validation** — extend KHOJ: neighbors' positions disagree with your GPS = spoof.
- **Autonomous response** — on detection, drop GPS → inertial/VIO nav → return-to-home.

---

## Section D — Metrics to report (this is what makes it credible)
- **Detection rate** (% of spoof events caught)
- **False-alarm rate** (flags during clean flight — keep low)
- **Detection latency** (seconds from spoof onset to flag)
- **Breakdown by spoof type** (jump / slow-drift / replay — drift is the headline)

---

## Section E — Deliverables that make it land
1. **GitHub repo** — clean README, architecture diagram, result plots, honest "Limitations & Future Work."
2. **60–90 s demo video** — the detector catching a spoof (video >> text for cold outreach).
3. **One-page technical note** — frame it in the counter-UAS / EW context.
4. **Resume one-liner** (fill the blanks after Phase 2–3):
   `GNSS Spoofing Detection Layer (ESP32, IMU, u-blox GPS): cross-checks GNSS against inertial dead-reckoning (EKF innovation monitoring) to detect/reject spoofed fixes; validated on injected attacks [+ TEXBAT]; ~__ s detection latency at <__% false alarms.`

---

## Section F — README template (drop into the repo)
```markdown
# GNSS Spoofing Detection Layer
Detecting GPS spoofing on UAVs by cross-checking GNSS against inertial/visual odometry.

## Why
Jamming denies signal; spoofing feeds a *false* position the drone trusts. Spoofing is
subtler, more dangerous, and far less solved — a key counter-UAS / EW problem.

## Approach
[architecture diagram]
GNSS fix + IMU → dead-reckoning / EKF → innovation & consistency monitor → spoof decision.

## Results
- Detection rate: __%   False-alarm rate: __%   Detection latency: __ s
- [ROC curve]  [latency plot]
- Spoof types tested: jump / slow-drift / replay

## Run
`python -m eval.run --scenario drift`

## Limitations & Future Work
Proof of concept; simulated/injected spoofing (no RF transmission). Next: TEXBAT
benchmark, VIO cross-check, swarm cross-validation, autonomous fail-safe response.
```
