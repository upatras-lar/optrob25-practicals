import numpy as np
import settings

def angle_dist(b, a = 0.):
    theta = b - a
    while theta < -np.pi:
        theta += 2. * np.pi
    while theta > np.pi:
        theta -= 2. * np.pi
    return theta

def hat(vec):
    v = vec.reshape((3,))
    return np.array([
        [0., -v[2], v[1]],
        [v[2], 0., -v[0]],
        [-v[1], v[0], 0.]
    ])

def unhat(mat):
    return np.array([[mat[2, 1], mat[0, 2], mat[1, 0]]]).T

def J_r(q, epsilon = 1e-8):
    n = np.linalg.norm(q)
    if n < epsilon:
        return np.eye(3)
    n_sq = n * n
    n_3 = n_sq * n
    c = np.cos(n)
    s = np.sin(n)
    hat_q = hat(q)
    hat_q_sq = hat_q @ hat_q
    JR = np.eye(3) - hat_q * ((1. - c) / n_sq) + hat_q_sq * ((n - s) / n_3)
    return JR

def J_r_inv(q, epsilon = 1e-8):
    n = np.linalg.norm(q)
    if n < epsilon:
        return np.eye(3)
    n_sq = n * n
    n_3 = n_sq * n
    c = np.cos(n)
    s = np.sin(n)
    hat_q = hat(q)
    hat_q_sq = hat_q @ hat_q
    JR_inv = np.eye(3) + 0.5 * hat_q + hat_q_sq * (1./n_sq - (1. + c) / (2. * n * s))
    return JR_inv

def J_l(q, epsilon = 1e-8):
    n = np.linalg.norm(q)
    if n < epsilon:
        return np.eye(3)
    n_sq = n * n
    n_3 = n_sq * n
    c = np.cos(n)
    s = np.sin(n)
    hat_q = hat(q)
    hat_q_sq = hat_q @ hat_q
    JL = np.eye(3) + hat_q * ((1. - c) / n_sq) + hat_q_sq * ((n - s) / n_3)
    return JL

def J_l_inv(q, epsilon = 1e-8):
    n = np.linalg.norm(q)
    if n < epsilon:
        return np.eye(3)
    n_sq = n * n
    n_3 = n_sq * n
    c = np.cos(n)
    s = np.sin(n)
    hat_q = hat(q)
    hat_q_sq = hat_q @ hat_q
    JL_inv = np.eye(3) - 0.5 * hat_q + hat_q_sq * (1./n_sq - (1. + c) / (2. * n * s))
    return JL_inv

def deriv_hat_vec(skew, vec):
    s = skew.reshape((3,))
    v = vec.reshape((3,))

    # out = skew * v
    # out[0] = -s[2]*v[1] + s[1]*v[2]
    # out[1] = s[2]*v[0] - s[0]*v[2]
    # out[2] = -s[1]*v[0] + s[0]*v[1]

    jac = np.zeros((3, 3))
    # out[0] wrt s[0]
    # jac[0, 0] = 0.
    # out[0] wrt s[1]
    jac[0, 1] = v[2]
    # out[0] wrt s[2]
    jac[0, 2] = -v[1]

    # out[1] wrt s[0]
    jac[1, 0] = -v[2]
    # out[1] wrt s[1]
    # jac[1, 1) = 0.
    # out[1] wrt s[2]
    jac[1, 2] = v[0]

    # out[2] wrt s[0]
    jac[2, 0] = v[1]
    # out[2] wrt s[1]
    jac[2, 1] = -v[0]
    # out[2] wrt s[2]
    # jac[2, 2] = 0.

    return jac

def exp_rotation(p, epsilon = 1e-8):
    phi = p.reshape((3, 1))
    theta = np.linalg.norm(phi)

    if theta < epsilon:
        return np.eye(3, 3)
    a = phi / theta

    c = np.cos(theta)
    s = np.sin(theta)

    return np.eye(3) * c + (1. - c) * a @ a.T + s * hat(a)

def deg2rad(degrees):
    return np.pi / 180. * degrees

def log_rotation(R, eps = 1e-8):
    theta = np.arccos(max(-1., min(1., (np.trace(R) - 1.) / 2.)))

    if np.isclose(theta, 0., rtol=eps, atol=eps):
        return np.zeros((3, 1))
    elif np.isclose(theta, np.pi, rtol=eps, atol=eps):
        r00 = R[0, 0]
        r11 = R[1, 1]
        r22 = R[2, 2]

        r02 = R[0, 2]
        r12 = R[1, 2]

        r01 = R[0, 1]
        r21 = R[2, 1]

        r10 = R[1, 0]
        r20 = R[2, 0]

        if not np.isclose(r22, -1., rtol=eps, atol=eps):
            multiplier = theta / np.sqrt(2. * (1. + r22))
            return multiplier * np.array([[r02, r12, 1. + r22]]).T
        elif not np.isclose(r11, -1., rtol=eps, atol=eps):
            multiplier = theta / np.sqrt(2. * (1. + r11))
            return multiplier * np.array([[r01, 1. + r11, r21]]).T
        elif not np.isclose(r00, -1., rtol=eps, atol=eps):
            multiplier = theta / np.sqrt(2. * (1. + r00))
            return multiplier * np.array([[1. + r00, r10, r20]]).T
        # else:
        #     print()
        #     print("This can't happen!")
        #     print(R)
        #     exit(1)

    mat = R - R.T
    r = unhat(mat)

    return theta / (2. * np.sin(theta)) * r

def Rx(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.array([
        [1., 0., 0.],
        [0., ct, -st],
        [0., st, ct]
    ])
    return R

def Ry(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.array([
        [ct, 0., st],
        [0., 1., 0.],
        [-st, 0., ct]
    ])
    return R

def Rz(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.array([
        [ct, -st, 0.],
        [st, ct, 0.],
        [0., 0., 1.]
    ])
    return R

def quaternion_to_rotation_matrix(q):
    if settings.NO_QUAT_NORMALIZATION:
        Q = q.reshape((4,))
    else:
        Q = q.reshape((4,)) / np.linalg.norm(q)
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2. * (q0 * q0 + q1 * q1) - 1.
    r01 = 2. * (q1 * q2 - q0 * q3)
    r02 = 2. * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2. * (q1 * q2 + q0 * q3)
    r11 = 2. * (q0 * q0 + q2 * q2) - 1.
    r12 = 2. * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2. * (q1 * q3 - q0 * q2)
    r21 = 2. * (q2 * q3 + q0 * q1)
    r22 = 2. * (q0 * q0 + q3 * q3) - 1.

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

def aa_to_quat(aa, epsilon = 1e-8):
    angle = np.linalg.norm(aa)
    v = aa / (angle + epsilon)
    s = np.sin(angle / 2.)
    return np.array([[np.cos(angle / 2.), v[0, 0]  * s, v[1, 0]  * s, v[2, 0]  * s]]).T

def rotation_matrix_to_euler_zyx(R):
    X, Y, Z = 2, 1, 0
    euler = np.zeros((3, 1))
    R20 = R[2, 0]

    if R20 < 1:
        if R20 > -1:
            euler[Y, 0] = np.arcsin(-R20)
            euler[Z, 0] = np.arctan2(R[1, 0], R[0, 0])
            euler[X, 0] = np.arctan2(R[2, 1], R[2, 2])
        else:  # R20 == -1, not a unique solution
            euler[Y, 0] = np.pi / 2
            euler[Z, 0] = -np.arctan2(-R[1, 2], R[1, 1])
            euler[X, 0] = 0
    else:  # R20 == 1, not a unique solution
        euler[Y, 0] = -np.pi / 2
        euler[Z, 0] = np.arctan2(-R[1, 2], R[1, 1])
        euler[X, 0] = 0

    return euler

def log_quat(q, epsilon = 1e-8):
    qw = q[0, 0]
    s = np.linalg.norm(q[1:]) + epsilon

    return 2. * q[1:, :] * np.arctan2(s, qw) / s

def jac_log_quat(q, epsilon = 1e-8):
    qw = q[0, 0]
    v = np.sum(np.square(q[1:])) + epsilon
    s = np.sqrt(np.sum(np.square(q[1:]))) + epsilon
    s_sq = s * s

    datan2_dqw = -s / (qw * qw + s_sq)
    datan2_ds = qw / (qw * qw + s_sq)
    done_s_dq = -(q[1:, 0] / np.power(v, 1.5)).T
    ds_dq = (q[1:, 0] / s).T

    J = np.zeros((3, 4))
    J[:, 0:1] = 2. * q[1:, :] * datan2_dqw / s
    J[:, 1:] = 2.* (np.eye(3) * np.arctan2(s, qw) / s + q[1:, :] * np.arctan2(s, qw) * done_s_dq + q[1:, :] / s * datan2_ds * ds_dq)

    return J
