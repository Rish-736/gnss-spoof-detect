"""Attitude maths: quaternions, rotations, and quadrotor flatness.

Conventions, fixed once here and obeyed everywhere else
-------------------------------------------------------
Frames
    nav  : local-level **NED** -- x North, y East, z **Down**.
           Gravity is therefore g_nav = [0, 0, +9.81]: positive, because down
           is positive. Getting this sign wrong flips every vertical channel
           and the error looks exactly like a 2 g accelerometer bias.
    body : x forward, y right, z down (standard aerospace).

Quaternion
    q = [w, x, y, z], unit norm, representing the **body-to-nav** rotation:
    v_nav = R(q) @ v_body. Scalar-first ordering is the aerospace convention;
    SciPy and Eigen use scalar-last, so anything crossing that boundary needs
    reordering. We do not cross it.

Specific force
    An accelerometer does not measure acceleration -- it measures specific
    force, the non-gravitational part:

        f_body = R(q)^T @ (a_nav - g_nav)

    In free fall it reads zero; sitting on a bench it reads +1 g upward. This
    is the single most important equation in the file, because undoing it is
    the first thing the detector must do, and any attitude error in R leaks
    gravity straight into the horizontal channels. One degree of tilt error
    injects 9.81 * sin(1 deg) = 0.17 m/s^2 of phantom horizontal acceleration
    -- forty times the accelerometer's own noise. That is why Phase 1 is
    mostly a fight about attitude, not about position.
"""

from __future__ import annotations

import numpy as np

#: Standard gravity in the NED navigation frame [m/s^2]; +z is down.
G_NED = np.array([0.0, 0.0, 9.80665])


# ---------------------------------------------------------------------------
# quaternion basics
# ---------------------------------------------------------------------------

def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Renormalise, guarding the degenerate case.

    Numerical drift makes |q| wander from 1; left alone it shows up as a slow
    scaling of every rotated vector. Cheap to fix, so it is fixed every step.
    """
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return quat_identity()
    return q / n


def quat_mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a (x) b, scalar-first."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Body-to-nav rotation matrix: v_nav = R @ v_body."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Inverse of `quat_to_rot`, via Shepperd's method.

    The naive formula divides by w, which blows up near 180 degrees of
    rotation. Shepperd picks whichever of the four components is largest and
    solves from that, so it is stable everywhere.
    """
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s,
                      0.25 * s,
                      (R[0, 1] + R[1, 0]) / s,
                      (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s,
                      (R[0, 1] + R[1, 0]) / s,
                      0.25 * s,
                      (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s,
                      (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s,
                      0.25 * s])
    return quat_normalize(q)


def quat_integrate(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    """Propagate attitude by a body-frame angular rate over `dt`.

    Uses the exact exponential map rather than the usual first-order
    `q += 0.5 * Omega q dt`. For constant omega this is exact to machine
    precision, where first order leaves an O(dt^2) error that integrates into
    heading drift -- the same class of mistake that cost 17 m in Phase 0.
    """
    w = np.asarray(omega_body, float)
    theta = np.linalg.norm(w) * dt
    if theta < 1e-12:
        return quat_normalize(q)
    axis = w / np.linalg.norm(w)
    half = 0.5 * theta
    dq = np.concatenate([[np.cos(half)], np.sin(half) * axis])
    return quat_normalize(quat_mult(q, dq))


def quat_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest rotation angle [rad] taking `a` to `b` -- the natural error
    metric for attitude, and the one that maps directly onto leaked gravity."""
    d = quat_mult(quat_conj(quat_normalize(a)), quat_normalize(b))
    return float(2.0 * np.arctan2(np.linalg.norm(d[1:]), abs(d[0])))


# ---------------------------------------------------------------------------
# Euler angles -- diagnostics and reporting only, never for propagation
# ---------------------------------------------------------------------------

def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Return (roll, pitch, yaw) in radians, aerospace 3-2-1 sequence.

    For humans and plots. Attitude is never *propagated* through Euler angles:
    pitching to +/-90 degrees makes roll and yaw degenerate (gimbal lock), and
    a quadrotor in an aggressive manoeuvre gets closer to that than you would
    like. Quaternions have no such singularity.
    """
    w, x, y, z = quat_normalize(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(s)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return quat_normalize(np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]))


# ---------------------------------------------------------------------------
# quadrotor differential flatness
# ---------------------------------------------------------------------------

def attitude_from_accel_yaw(a_nav: np.ndarray, yaw: float) -> np.ndarray:
    """The attitude a quadrotor **must** hold to produce acceleration `a_nav`.

    A multirotor has exactly one actuator direction: thrust along body -z. So

        m * a_nav = m * g_nav + R @ [0, 0, -T]

    which forces the body z-axis in nav coordinates to be

        b3 = unit(g_nav - a_nav)

    Yaw is the remaining free parameter. This is the differential-flatness
    property of quadrotors (Mellinger & Kumar): pick a position trajectory and
    a yaw, and the attitude is determined, not chosen.

    What `yaw` means precisely: the construction guarantees **body-y is
    perpendicular to the horizontal heading vector** [cos(yaw), sin(yaw), 0],
    exactly. It does *not* guarantee that the ZYX Euler yaw of the result
    equals `yaw` -- once the vehicle tilts those two differ, by up to ~20
    degrees at 45 degrees of bank. Both are legitimate definitions of heading
    for a tilted body; this one is the standard flatness convention and is
    exact. Nothing downstream depends on the distinction, because the detector
    consumes the full attitude rather than a yaw angle, but a test comparing
    Euler yaw against the commanded value will look wrong when it is not.

    Why it matters here: it means the accelerometer of a quadrotor reads
    essentially [0, 0, -T/m] -- almost nothing on the horizontal axes. All the
    horizontal motion information lives in the **attitude**, which is why
    horizontal navigation accuracy is governed by attitude accuracy and why
    Phase 1 is really a fight about tilt.
    """
    a_nav = np.asarray(a_nav, float)
    b3 = G_NED - a_nav
    n = np.linalg.norm(b3)
    if n < 1e-9:           # free fall: thrust direction undefined, hold level
        b3 = np.array([0.0, 0.0, 1.0])
    else:
        b3 = b3 / n

    # Heading reference in the horizontal plane.
    c1 = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    b2 = np.cross(b3, c1)
    n2 = np.linalg.norm(b2)
    if n2 < 1e-9:          # pointing straight down the heading vector
        c1 = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        b2 = np.cross(b3, c1)
        n2 = np.linalg.norm(b2)
    b2 = b2 / n2
    b1 = np.cross(b2, b3)
    return rot_to_quat(np.column_stack([b1, b2, b3]))


def rates_from_attitude(quats: np.ndarray, dt: float) -> np.ndarray:
    """Body angular rates implied by an attitude sequence.

    Recovered from the relative rotation between consecutive samples via the
    log map, so the result is consistent with `quat_integrate` by construction
    -- feed these rates back in and you get the same attitudes out. Phase 0
    taught us that truth generated by one scheme and consumed by another
    leaves a systematic error that masquerades as sensor bias.
    """
    quats = np.asarray(quats, float)
    n = len(quats)
    omega = np.zeros((n, 3))
    for i in range(n - 1):
        d = quat_mult(quat_conj(quats[i]), quats[i + 1])
        if d[0] < 0.0:
            d = -d                      # shortest path
        v = d[1:]
        s = np.linalg.norm(v)
        if s < 1e-12:
            continue
        angle = 2.0 * np.arctan2(s, d[0])
        omega[i] = (angle / dt) * (v / s)
    omega[-1] = omega[-2] if n > 1 else 0.0
    return omega
