import numpy as np

from spaces import VectorSpace, CombinedSpace, UnitQuaternionSpace, SO3Space

from math_utils import hat, deriv_hat_vec

class QuadrotorEnv:
    def __init__(self, orientation, tf = 10., dt = 0.05, hard_control_limits = False, end_bad_episodes = True):
        self._m = 0.033 # mass of the quadrotor in kg
        self._l = 0.046 # length of the quadrotor arm in m
        self._I = np.array([[16.6e-6, 0.83e-6, 0.72e-6],
                            [0.83e-6, 16.6e-6, 1.8e-6],
                            [0.72e-6, 1.8e-6, 29.3e-6]])
        self._I_inv = np.linalg.inv(self._I)

        self._g = 9.81
        self._yaw_zero = -3/4 * np.pi  # assuming body_yaw0 is for the motor 1 at positive y direction, motor 2 at positive x direction and clockwise motor numbers
        self._Kf = 2.25e-8  # thrust coefficient in N/(rad/s)^2
        self._Kt = 1.34e-10  # torque coefficient in N*m/(rad/s)^2
        self._Km = self._Kt / self._Kf

        self._dt = dt
        self._tf = tf
        self._K = round(self._tf / self._dt) + 1

        self._space = CombinedSpace([VectorSpace(3), orientation])
        self._full_space = CombinedSpace([VectorSpace(3), orientation, VectorSpace(3), VectorSpace(3)])
        self._quat_trick = (type(orientation) is UnitQuaternionSpace) and orientation.nd() == 3
        self._so3 = (type(orientation) is SO3Space)

        self._hard_control_limits = hard_control_limits
        self._end_bad_episodes = end_bad_episodes

        self._nu = 4

    def nq(self):
        return self._space._nq + self._space._nv

    def nv(self):
        return self._space._nv + self._space._nv

    def nd(self):
        return self._space.nd() + self._space._nv

    def _E(self, x):
        nq = self._space._nq
        nv = self._space._nv

        E = np.zeros((nq + nv, nv + nv))
        E[:3, :3] = np.eye(3)
        E[3:7, 3:6] = self._space._spaces[-1].G(x[3:7, :])
        E[7:10, 6:9] = np.eye(3)
        E[10:, 9:] = np.eye(3)
        return E

    def apply_control_limits(self, u):
        if self._hard_control_limits:
            lims = self.control_limits()
            return np.maximum(np.minimum(u, lims[0, 1]), lims[0, 0])
        return u

    def control_limits(self):
        if self._hard_control_limits:
            return np.array([[0., 2. * self._m * self._g]])
        return np.array([[-np.inf, np.inf]])

    def step(self, x, u):
        nq = self._space._nq
        nv = self._space._nv
        dt = self._dt

        pos = x[:nq]
        vel = x[nq:]

        _, rot, v, omega = self._full_space.expand(x)

        # Let's compute accelerations
        m = self._m
        l = self._l
        I = self._I
        I_inv = self._I_inv
        g = self._g
        Km = self._Km

        Kmat = np.array([
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [1., 1., 1., 1.]
        ])

        if self._hard_control_limits:
            u = self.apply_control_limits(u)

        lin_acc = rot.T @ np.array([[0., 0., -g]]).T + ((1. / m) * Kmat) @ u - hat(omega) @ v

        cy = np.cos(self._yaw_zero)
        sy = np.sin(self._yaw_zero)
        tmp = np.array([
            [l * cy, l * sy, -l * cy, -l * sy],
            [l * sy, -l * cy, -l * sy, l * cy],
            [-Km, Km, -Km, Km]
        ])
        ang_acc = I_inv @ (-hat(omega) @ I @ omega + tmp @ u)

        # Concatenate linear and angular parts
        acc = np.concatenate([lin_acc, ang_acc])

        # Semi-implicit Euler
        vn = vel + acc * dt
        # transform linear velocity to world-space velocity
        R = np.block([[rot, np.zeros((3, 3))], [np.zeros((3, 3)), np.eye(3)]])

        xn = self._space.step(pos, R @ vn, dt)

        if self._end_bad_episodes and ((np.abs(vn) > 100.).any() or (np.abs(xn[:3]) > 10.).any()):
            return x

        return np.concatenate([xn, vn])

    def A(self, x, u, xn = None):
        nq = self._space._nq
        nv = self._space._nv
        nd = self._space.nd()
        dt = self._dt

        pos = x[:nq]
        vel = x[nq:]

        q = x[3:nq]

        _, rot, v, omega = self._full_space.expand(x)

        # Let's compute the forward step first
        m = self._m
        l = self._l
        I = self._I
        I_inv = self._I_inv
        g = self._g
        Km = self._Km

        Kmat = np.array([
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [1., 1., 1., 1.]
        ])

        if self._hard_control_limits:
            u = self.apply_control_limits(u)

        lin_acc = rot.T @ np.array([[0., 0., -g]]).T + ((1. / m) * Kmat) @ u - hat(omega) @ v

        cy = np.cos(self._yaw_zero)
        sy = np.sin(self._yaw_zero)
        tmp = np.array([
            [l * cy, l * sy, -l * cy, -l * sy],
            [l * sy, -l * cy, -l * sy, l * cy],
            [-Km, Km, -Km, Km]
        ])
        ang_acc = I_inv @ (-hat(omega) @ I @ omega + tmp @ u)

        # Concatenate linear and angular parts
        acc = np.concatenate([lin_acc, ang_acc])

        # Semi-implicit Euler
        vn = vel + acc * dt
        # transform linear velocity to world-space velocity
        R = np.block([[rot, np.zeros((3, 3))], [np.zeros((3, 3)), np.eye(3)]])

        if self._so3:
            nq = nd

        J = np.zeros((nq + nv, nq + nv))

        # xn = f(z), in R^nq
        # z = [x h(x, vn)], in R^(nq+nv)
        # vn = v + g(x, v) * dt, in R^nv
        # h(x, vn) = R(x) @ vn, in R^nv
        # g(x, v) = acc, in R^nv

        # dpos -> df/dz, in R^(nq*(nq+nv))
        dpos = np.zeros((nq, nq + nv))
        dpos[:, :nq] = self._space.step_deriv(pos, R @ vn, dt, arg = 0, so3 = self._so3)
        dpos[:, nq:] = self._space.step_deriv(pos, R @ vn, dt, arg = 1, so3 = self._so3)

        # dh_dvn -> dh/dvn, in R^(nv*nv)
        dh_dvn = R

        # dg_dx -> dg/dx, in R^(nv*nq)
        dg_dx = np.block([[np.zeros((3, 3)), self._space._spaces[-1].deriv_rot_transpose_vec(q, np.array([[0., 0., -g]]).T)], [np.zeros((3, nq))]])

        # dh_dx -> dh/dx, in R^(nv*nq)
        dh_dx = np.block([[np.zeros((3, 3)), self._space._spaces[-1].deriv_rot_vec(q, vn[:3])], [np.zeros((3, nq))]]) + R @ dg_dx * dt

        # dg_dv -> dg/dv, in R^(nv*nv)
        dg_dv = np.block([[-hat(omega), -deriv_hat_vec(omega, v)], [np.zeros((3, 3)), I_inv @ (-deriv_hat_vec(omega, I @ omega) - hat(omega) @ I)]])

        # dvn_dv -> dvn/dv, in R^(nv*nv)
        dvn_dv = np.eye(nv) + dg_dv * dt

        # dz_dx -> dz/dx, in R^((nq+nv)*nq)
        dz_dx = np.block([[np.eye(nq)], [dh_dx]])

        # dz_dvn -> dz/dvn, in R^((nq+nv)*nv)
        dz_dvn = np.block([[np.zeros((nq, nv))], [dh_dvn]])

        # dpos/dq
        J[:nq, :nq] = dpos @ dz_dx
        # dpos/dv
        J[:nq, nq:] = dpos @ dz_dvn @ dvn_dv
        # dvel/dq
        J[nq:, :nq] = dg_dx * dt
        # dvel/dv
        J[nq:, nq:] = dvn_dv

        if self._quat_trick:
            assert(xn is not None)
            En = self._E(xn)
            E = self._E(x)
            J = En.T @ J @ E

        return J

    def B(self, x, u, xn = None):
        nq = self._space._nq
        nv = self._space._nv
        nd = self._space.nd()
        nu = 4
        dt = self._dt

        pos = x[:nq]
        vel = x[nq:]

        q = x[3:nq]

        _, rot, v, omega = self._full_space.expand(x)

        # Let's compute the forward step first
        m = self._m
        l = self._l
        I = self._I
        I_inv = self._I_inv
        g = self._g
        Km = self._Km

        Kmat = np.array([
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [1., 1., 1., 1.]
        ])

        if self._hard_control_limits:
            u = self.apply_control_limits(u)

        lin_acc = rot.T @ np.array([[0., 0., -g]]).T + ((1. / m) * Kmat) @ u - hat(omega) @ v

        cy = np.cos(self._yaw_zero)
        sy = np.sin(self._yaw_zero)
        tmp = np.array([
            [l * cy, l * sy, -l * cy, -l * sy],
            [l * sy, -l * cy, -l * sy, l * cy],
            [-Km, Km, -Km, Km]
        ])
        ang_acc = I_inv @ (-hat(omega) @ I @ omega + tmp @ u)

        # Concatenate linear and angular parts
        acc = np.concatenate([lin_acc, ang_acc])

        # Semi-implicit Euler
        vn = vel + acc * dt
        # transform linear velocity to world-space velocity
        R = np.block([[rot, np.zeros((3, 3))], [np.zeros((3, 3)), np.eye(3)]])

        if self._so3:
            nq = nd

        J = np.zeros((nq + nv, nu))

        # xn = f(z), in R^nq
        # z = [x h(vn)], in R^(nq+nv)
        # vn = v + g(u) * dt, in R^nv
        # h(u) = R @ vn(u), in R^nv
        # g(u) = acc, in R^nv

        # dv = df/dz @ dz/dh (we are basically selecting only the v derivative; no need to compute it and remove it)
        dv = self._space.step_deriv(pos, R @ vn, dt, arg = 1, so3 = self._so3)

        # dh_dvn -> dh/dvn, in R^nv*nv
        dh_dvn = R

        # dvn_du -> dvn/du, in R^nv*nu
        dvn_du = np.block([[(1. / m) * Kmat], [I_inv @ tmp]]) * dt

        # dpos/du
        J[:nq, :] = dv @ dh_dvn @ dvn_du
        # dvel/du
        J[nq:, :] = dvn_du

        if self._quat_trick:
            assert(xn is not None)
            En = self._E(xn)
            J = En.T @ J

        return J
