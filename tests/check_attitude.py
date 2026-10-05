"""Sanity checks on the attitude core, before anything is built on it."""
import numpy as np
from sim import attitude as A

rng = np.random.default_rng(0)
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print("  %-46s %s  %s" % (name, "PASS" if cond else "FAIL", detail))


print("1. quaternion <-> rotation round trip")
err = 0.0
for _ in range(200):
    q = A.quat_normalize(rng.normal(size=4))
    if q[0] < 0:
        q = -q
    q2 = A.rot_to_quat(A.quat_to_rot(q))
    if q2[0] < 0:
        q2 = -q2
    err = max(err, np.abs(q - q2).max())
check("round trip error", err < 1e-9, "max %.2e" % err)

print("\n2. rotation matrices are orthonormal, det +1")
e1 = e2 = 0.0
for _ in range(200):
    R = A.quat_to_rot(A.quat_normalize(rng.normal(size=4)))
    e1 = max(e1, np.abs(R @ R.T - np.eye(3)).max())
    e2 = max(e2, abs(np.linalg.det(R) - 1.0))
check("orthonormality / determinant", e1 < 1e-12 and e2 < 1e-12,
      "%.1e / %.1e" % (e1, e2))

print("\n3. yaw-only rotation reproduces the Phase 0 2-D matrix")
psi = 0.7
R = A.quat_to_rot(A.euler_to_quat(0, 0, psi))
expect = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
check("matches 2-D convention", np.abs(R - expect).max() < 1e-12)

print("\n4. Euler round trip (away from gimbal lock)")
err = 0.0
for _ in range(200):
    r, p, y = rng.uniform(-np.pi, np.pi), rng.uniform(-1.2, 1.2), rng.uniform(-np.pi, np.pi)
    e = A.quat_to_euler(A.euler_to_quat(r, p, y))
    d = np.array([r, p, y]) - e
    d = (d + np.pi) % (2 * np.pi) - np.pi
    err = max(err, np.abs(d).max())
check("euler round trip", err < 1e-9, "max %.2e rad" % err)

print("\n5. quat_integrate is exact for constant rate")
w = np.array([0.3, -0.2, 0.5])
dt, n = 0.01, 1000
q = A.quat_identity()
for _ in range(n):
    q = A.quat_integrate(q, w, dt)
q_exact = A.quat_identity()
q_exact = A.quat_integrate(q_exact, w, dt * n)   # one big step == exact
err = A.quat_angle_between(q, q_exact)
check("1000 steps == 1 step", err < 1e-9, "%.2e rad" % err)

print("\n6. specific force: at rest the accelerometer reads +1 g up")
q = A.quat_identity()                      # level
f = A.quat_to_rot(q).T @ (np.zeros(3) - A.G_NED)
check("level, stationary -> [0,0,-9.81]", np.allclose(f, [0, 0, -9.80665]),
      str(np.round(f, 3)))
f_ff = A.quat_to_rot(q).T @ (A.G_NED - A.G_NED)
check("free fall -> zero", np.allclose(f_ff, 0))

print("\n7. flatness: attitude -> required accel -> same attitude")
worst_a, worst_tilt = 0.0, 0.0
for _ in range(300):
    a_nav = np.array([rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-3, 3)])
    yaw = rng.uniform(-np.pi, np.pi)
    q = A.attitude_from_accel_yaw(a_nav, yaw)
    R = A.quat_to_rot(q)
    # thrust magnitude implied, then the acceleration it would produce
    T = np.linalg.norm(A.G_NED - a_nav)
    a_back = A.G_NED + R @ np.array([0.0, 0.0, -T])
    worst_a = max(worst_a, np.abs(a_back - a_nav).max())
    # The construction's exact guarantee is body-y _|_ heading vector, NOT
    # that the Euler yaw equals the commanded yaw (those diverge once tilted).
    c1 = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    worst_tilt = max(worst_tilt, abs(float(R[:, 1] @ c1)))
check("acceleration reproduced", worst_a < 1e-9, "max %.2e m/s^2" % worst_a)
check("body-y perpendicular to heading", worst_tilt < 1e-12, "max %.2e" % worst_tilt)

print("\n8. hover demands level attitude; 1 m/s^2 north demands ~5.8 deg pitch")
q_hover = A.attitude_from_accel_yaw(np.zeros(3), 0.0)
check("hover is level", A.quat_angle_between(q_hover, A.quat_identity()) < 1e-9)
q_acc = A.attitude_from_accel_yaw(np.array([1.0, 0, 0]), 0.0)
pitch = np.degrees(A.quat_to_euler(q_acc)[1])
expect = np.degrees(np.arctan2(1.0, 9.80665))
check("pitch for 1 m/s^2", abs(pitch + expect) < 1e-6,
      "%.3f deg (expect %.3f, nose-down is -ve)" % (pitch, -expect))

print("\n9. rates_from_attitude inverts quat_integrate")
dt = 0.01
qs = [A.quat_identity()]
true_w = []
for i in range(500):
    w = np.array([0.4 * np.sin(i * dt), 0.3 * np.cos(0.7 * i * dt), 0.2])
    true_w.append(w)
    qs.append(A.quat_integrate(qs[-1], w, dt))
qs = np.array(qs[:-1])
rec = A.rates_from_attitude(qs, dt)
err = np.abs(rec[:-2] - np.array(true_w)[:-2]).max()
check("recovered rates match", err < 1e-9, "max %.2e rad/s" % err)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
raise SystemExit(0 if ok else 1)
