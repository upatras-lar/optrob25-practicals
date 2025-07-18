import numpy as np
import copy
import settings
from math_utils import hat, exp_rotation, log_rotation, J_r, J_r_inv, J_l, J_l_inv, log_quat, jac_log_quat, quaternion_to_rotation_matrix, aa_to_quat, Rx, Ry, Rz

class VectorSpace:
    def __init__(self, N):
        self._nq = N
        self._nv = N

    def nd(self):
        return self._nq

    def zero(self):
        return np.zeros((self._nq, 1))

    def expand(self, x):
        return x

    def state_diff(self, target, current):
        return target - current

    def state_diff_deriv(self, target, current, arg = 1):
        if arg == 0:
            return np.eye(target.shape[0])
        return -np.eye(current.shape[0])

    def step(self, x, v, dt):
        # Euler integration
        return x + v * dt

    def step_deriv(self, x, v, dt, arg = 0, so3 = False):
        nv = self._nv
        # xn = x + v * dt
        if arg == 0: # over x
            return np.eye(nv) # dxn/dx
        # over v
        return np.eye(nv) * dt # dxn/dv

    def to(self, x, Type):
        if Type == VectorSpace:
            return x
        assert(False and "Invalid space conversion!")

class SO3Space:
    def __init__(self):
        self._nq = 9 # 3x3 rotation matrices
        self._nv = 3

    def nd(self):
        return self._nv

    def zero(self):
        return np.eye(3).reshape((-1, 1))

    def expand(self, x):
        return x.reshape((3, 3))

    def state_diff(self, target, current):
        Rt = target.reshape((3, 3))
        R = current.reshape((3, 3))
        return log_rotation(R.T @ Rt)

    def state_diff_deriv(self, target, current, arg = 1):
        tau = self.state_diff(target, current)
        if arg == 0:
            return J_r_inv(tau)
        return -J_l_inv(tau)
        # Rt = target.reshape((3, 3))
        # R = current.reshape((3, 3))
        # Rd = R.T @ Rt
        # tau = self.state_diff(target, current)
        # if arg == 0:
        #     return J_r_inv(tau)
        # return -J_r(tau) @ Rd.T

    def step(self, x, v, dt):
        # Euler integration
        R = x.reshape((3, 3))
        return (R @ exp_rotation(v * dt)).reshape((-1, 1))

    def step_deriv(self, x, v, dt, arg = 0, so3 = False):
        if arg == 0: # over R
            return exp_rotation(-v * dt)
        # over v
        return J_r(v * dt) * dt

    def deriv_rot_vec(self, x, v):
        R = x.reshape((3, 3))
        return -R @ hat(v)

    def deriv_rot_transpose_vec(self, x, v):
        R = x.reshape((3, 3))
        return R.T @ hat(v) @ R

    def to(self, x, Type):
        if Type == AxisAngleSpace:
            return log_rotation(self.expand(x))
        elif Type == UnitQuaternionSpace:
            return aa_to_quat(log_rotation(self.expand(x)))
        elif Type == NaiveQuaternionSpace:
            return aa_to_quat(log_rotation(self.expand(x)))
        elif Type == SO3Space:
            return x
        assert(False and "Invalid space conversion!")

class AxisAngleSpace:
    def __init__(self):
        self._nq = 3
        self._nv = 3

    def nd(self):
        return self._nv

    def zero(self):
        return np.zeros((self._nq, 1))

    def expand(self, x):
        return exp_rotation(x)

    def state_diff(self, target, current):
        Rt = exp_rotation(target)
        R = exp_rotation(current)
        return log_rotation(R.T @ Rt)

    def state_diff_deriv(self, target, current, arg = 1):
        tau = self.state_diff(target, current)
        if arg == 0:
            return J_r_inv(tau) @ J_r(target)
        return -J_l_inv(tau) @ J_r(current)

    def step(self, x, v, dt):
        # Euler integration
        R = exp_rotation(x)
        return log_rotation(R @ exp_rotation(v * dt))

    def step_deriv(self, x, v, dt, arg = 0, so3 = False):
        tau = self.step(x, v, dt)
        if arg == 0: # over R
            return J_r_inv(tau) @ exp_rotation(-v * dt) @ J_r(x)
        # over v
        return J_r_inv(tau) @ J_r(v * dt) * dt

    def deriv_rot_vec(self, x, v):
        return -exp_rotation(x) @ hat(v) @ J_r(x)

    def deriv_rot_transpose_vec(self, x, v):
        return exp_rotation(x).T @ hat(v) @ J_r(-x)

    def to(self, x, Type):
        if Type == SO3Space:
            return self.expand(x).reshape((-1, 1))
        elif Type == UnitQuaternionSpace:
            return aa_to_quat(x)
        elif Type == NaiveQuaternionSpace:
            return aa_to_quat(x)
        elif Type == AxisAngleSpace:
            return x
        assert(False and "Invalid space conversion!")

class EulerAnglesSpace:
    def __init__(self):
        self._nq = 3
        self._nv = 3

    def nd(self):
        return self._nv

    def zero(self):
        return np.zeros((3, 1))

    def expand(self, x):
        return Rz(x[0, 0]) @ Ry(x[1, 0]) @ Rx(x[2, 0])

    def state_diff(self, target, current):
        return target - current

    def state_diff_deriv(self, target, current, arg = 1):
        if arg == 0:
            return np.eye(self._nq)
        return -np.eye(self._nq)

    def step(self, x, v, dt):
        dq = self.omega_to_dq(x, v)
        return x + dq * dt

    def step_deriv(self, x, v, dt, arg = 0, so3 = False):
        nq = self._nq
        if arg == 0: # over x
            return np.eye(nq) + self.dqdot_over_q(x, v) * dt # dxn/dx
        # over v
        return self.dqdot_over_omega(x, v) * dt # dxn/dv

    def omega_to_dq(self, q, omega):
        x = q[2, 0]
        y = q[1, 0]
        # z = q[0, 0]
        sy = np.sin(y)
        cy = np.cos(y)
        sx = np.sin(x)
        cx = np.cos(x)

        # E = np.array([[-sy, 0., 1.], [cy * sx, cx, 0.], [cx * cy, -sx, 0.]])
        Einv = np.array([[0., sx / cy, cx / cy], [0., cx, -sx], [1., sx * sy / cy, cx * sy / cy]])
        return Einv @ omega

    def dqdot_over_q(self, q, omega):
        x = q[2, 0]
        y = q[1, 0]
        # z = q[0, 0]
        sy = np.sin(y)
        cy = np.cos(y)
        sx = np.sin(x)
        cx = np.cos(x)

        omx = omega[0, 0]
        omy = omega[1, 0]
        omz = omega[2, 0]
        # dq[0] = omy * sx / cy + omz * cx / cy
        # dq[1] = omy * cx - omz * sx
        # dq[2] = omx + omy * sx * sy / cy + omz * cx * sy / cy
        jac = np.zeros((3, 3))
        # dq over z is all zeros
        # dq[0] over y
        jac[0, 1] = (omy * sx + omz * cx) * sy / (cy * cy)
        # dq[0] over x
        jac[0, 2] = omy * cx / cy - omz * sx / cy
        # dq[1] over y
        # jac[1, 1] = 0.
        # dq[1] over x
        jac[1, 2] = -omy * sx - omz * cx
        # dq[2] over y
        jac[2, 1] = omy * sx / (cy * cy) + omz * cx / (cy * cy)
        # dq[2] over x
        jac[2, 2] = omy * cx * sy / cy - omz * sx * sy / cy
        return jac

    def dqdot_over_omega(self, q, omega):
        x = q[2, 0]
        y = q[1, 0]
        # z = q[0, 0]
        sy = np.sin(y)
        cy = np.cos(y)
        sx = np.sin(x)
        cx = np.cos(x)

        # Einv!
        return np.array([[0., sx / cy, cx / cy], [0., cx, -sx], [1., sx * sy / cy, cx * sy / cy]])

    # d(R*v_b)/dq
    def deriv_rot_vec(self, q, v):
        x = q[2, 0]
        y = q[1, 0]
        z = q[0, 0]
        sz = np.sin(z)
        cz = np.cos(z)
        sy = np.sin(y)
        cy = np.cos(y)
        sx = np.sin(x)
        cx = np.cos(x)

        vx = v[0, 0]
        vy = v[1, 0]
        vz = v[2, 0]

        # out[0] = vx * cz * cy + vy * (sx * sy * cz - sz * cx) + vz * (sy * cx * cz + sx * sz)
        # out[1] = vx * sz * cy + vy * (sx * sz * sy + cx * cz) + vz * (sz * sy * cx - sx * cz)
        # out[2] = -vx * sy + vy * sx * cy + vz * cx * cy
        jac = np.zeros((3, 3))
        # out[0] over z
        jac[0, 0] = -vx * sz * cy + vy * (-sx * sy * sz - cz * cx) + vz * (-sy * cx * sz + sx * cz)
        # out[0] over y
        jac[0, 1] = -vx * cz * sy + vy * sx * cy * cz + vz * cy * cx * cz
        # out[0] over x
        jac[0, 2] = vy * (cx * sy * cz + sz * sx) + vz * (-sy * sx * cz + cx * sz)

        # out[1] over z
        jac[1, 0] = vx * cz * cy + vy * (sx * cz * sy - cx * sz) + vz * (cz * sy * cx + sx * sz)
        # out[1] over y
        jac[1, 1] = -vx * sz * sy + vy * sx * sz * cy + vz * sz * cy * cx
        # out[1] over x
        jac[1, 2] = vy * (cx * sz * sy - sx * cz) - vz * (sz * sy * sx + cx * cz)

        # out[2] over y
        jac[2, 1] = -vx * cy - vy * sx * sy - vz * cx * sy
        # out[2] over x
        jac[2, 2] = vy * cx * cy - vz * sx * cy
        return jac

    # d(R^T * v )/dq
    def deriv_rot_transpose_vec(self, q, v):
        x = q[2, 0]
        y = q[1, 0]
        z = q[0, 0]
        sz = np.sin(z)
        cz = np.cos(z)
        sy = np.sin(y)
        cy = np.cos(y)
        sx = np.sin(x)
        cx = np.cos(x)

        vx = v[0, 0]
        vy = v[1, 0]
        vz = v[2, 0]

        # out[0] = vx * cz * cy + vy * sz * cy - vz * sy
        # out[1] = vx * (sx * sy * cz - sz * cx) + vy * (sx * sz * sy + cx * cz) + vz * (sx * cy)
        # out[2] = vx * (sy * cx * cz + sx * sz) + vy * (sz * sy * cx - sx * cz) + vz * cx * cy
        jac = np.zeros((3, 3))
        # out[0] over z
        jac[0, 0] = -vx * sz * cy + vy * cz * cy
        # out[0] over y
        jac[0, 1] = -vx * cz * sy - vy * sz * sy - vz * cy

        # out[1] over z
        jac[1, 0] = vx * (-sx * sy * sz - cz * cx) + vy * (sx * cz * sy - cx * sz)
        # out[1] over y
        jac[1, 1] = vx * sx * cy * cz + vy * sx * sz * cy - vz * sx * sy
        # out[1] over x
        jac[1, 2] = vx * (cx * sy * cz + sz * sx) + vy * (cx * sz * sy - sx * cz) + vz * cx * cy

        # out[2] over z
        jac[2, 0] = vx * (-sy * cx * sz + sx * cz) + vy * (cz * sy * cx + sx * sz)
        # out[2] over y
        jac[2, 1] = vx * cy * cx * cz + vy * sz * cy * cx - vz * cx * sy
        # out[2] over x
        jac[2, 2] = vx * (-sy * sx * cz + cx * sz) + vy * (-sz * sy * sx - cx * cz) - vz * sx * cy
        return jac

    def to(self, x, Type):
        if Type == SO3Space:
            return self.expand(x).reshape((-1, 1))
        elif Type == AxisAngleSpace:
            return log_rotation(self.expand(x))
        elif Type == NaiveQuaternionSpace or Type == UnitQuaternionSpace:
            return aa_to_quat(log_rotation(self.expand(x)))
        elif Type == EulerAnglesSpace:
            return x
        assert(False and "Invalid space conversion!")

class UnitQuaternionSpace:
    def __init__(self, quat_trick = True):
        self._nq = 4
        self._nv = 3
        if quat_trick:
            self._nd = 3
        else:
            self._nd = 4

    def nd(self):
        return self._nd

    def zero(self):
        return np.array(([[1., 0., 0., 0.]])).T

    def expand(self, x):
        return quaternion_to_rotation_matrix(x) # exp_rotation(log_quat(x))

    def state_diff(self, target, current):
        if self._nd == 3:
            Rt = quaternion_to_rotation_matrix(target) # exp_rotation(log_quat(target))
            R = quaternion_to_rotation_matrix(current) # exp_rotation(log_quat(current))
            return log_rotation(R.T @ Rt)
        else:
            return target - current

    def state_diff_deriv(self, target, current, arg = 1):
        if self._nd == 3:
            qT = log_quat(target)
            qC = log_quat(current)
            Rt = quaternion_to_rotation_matrix(target) # exp_rotation(qT)
            R = quaternion_to_rotation_matrix(current) # exp_rotation(qC)
            tau = log_rotation(R.T @ Rt)
            if arg == 0:
                return J_r_inv(tau) @ J_r(qT) @ jac_log_quat(target) @ self.G(target)
            return -J_l_inv(tau) @ J_r(qC) @ jac_log_quat(current) @ self.G(current)
        else:
            if arg == 0:
                return np.eye(target.shape[0])
            return -np.eye(current.shape[0])

    def step(self, x, v, dt):
        nq = self._nq
        dq = self.omega_to_dq(x, v)
        # Euler integration
        xn = x + dq * dt
        if settings.NO_QUAT_NORMALIZATION:
            return xn
        return xn / np.linalg.norm(xn) # The norm messes up the finite diff grads but it's better for the rollouts!

    def step_deriv(self, x, v, dt, arg = 0, so3 = False):
        nq = self._nq
        # xn = x + dq * dt
        if arg == 0: # over x
            return np.eye(nq) + self.dqdot_over_q(x, v) * dt # dxn/dx
        # over v
        return self.dqdot_over_omega(x, v) * dt # dxn/dv

    def omega_to_dq(self, q, omega):
        om = np.concatenate([[[0.]], omega.reshape((3, 1))])
        return 0.5 * self.__L(q) @ om

    def dqdot_over_q(self, q, omega):
        return 0.5 * self.__R(self.__H() @ omega.reshape((3, 1)))

    def dqdot_over_omega(self, q, omega):
        return 0.5 * self.__L(q) @ self.__H()

    # d(R*v_b)/dq
    # https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    def deriv_rot_vec(self, qq, vv):
        q = qq.reshape((4,))
        v = vv.reshape((3,))

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        v0 = v[0]
        v1 = v[1]
        v2 = v[2]

        # out0 = r00*v0+r01*v1+r02*v2
        # out1 = r10*v0+r11*v1+r12*v2
        # out2 = r20*v0+r21*v1+r22*v2
        jac = np.zeros((3, 4))

        # dout0 / dq0
        jac[0, 0] = 4. * q0 * v0 - 2. * q3 * v1 + 2. * q2 * v2
        # dout0 / dq1
        jac[0, 1] = 4. * q1 * v0 + 2. * q2 * v1 + 2. * q3 * v2
        # dout0 / dq2
        jac[0, 2] = 2. * q1 * v1 + 2. * q0 * v2
        # dout0 / dq3
        jac[0, 3] = -2. * q0 * v1 + 2. * q1 * v2

        # dout1 / dq0
        jac[1, 0] = 2. * q3 * v0 + 4. * q0 * v1 - 2. * q1 * v2
        # dout1 / dq1
        jac[1, 1] = 2. * q2 * v0 - 2. * q0 * v2
        # dout1 / dq2
        jac[1, 2] = 2. * q1 * v0 + 4. * q2 * v1 + 2. * q3 * v2
        # dout1 / dq3
        jac[1, 3] = 2. * q0 * v0 + 2. * q2 * v2

        # dout2 / dq0
        jac[2, 0] = -2. * q2 * v0 + 2. * q1 * v1 + 4. * q0 * v2
        # dout1 / dq1
        jac[2, 1] = 2. * q3 * v0 + 2. * q0 * v1
        # dout1 / dq2
        jac[2, 2] = -2. * q0 * v0 + 2. * q3 * v1
        # dout1 / dq3
        jac[2, 3] = 2. * q1 * v0 + 2. * q2 * v1 + 4. * q3 * v2

        return jac

    # d(R^T * acc )/dq
    def deriv_rot_transpose_vec(self, qq, vv):
        q = qq.reshape((4,))
        v = vv.reshape((3,))

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        v0 = v[0]
        v1 = v[1]
        v2 = v[2]

        jac = np.zeros((3, 4))

        # dout0 / dq0
        jac[0, 0] = 4. * q0 * v0 + 2. * q3 * v1 - 2. * q2 * v2
        # dout0 / dq1
        jac[0, 1] = 4. * q1 * v0 + 2. * q2 * v1 + 2. * q3 * v2
        # dout0 / dq2
        jac[0, 2] = 2. * q1 * v1 - 2. * q0 * v2
        # dout0 / dq3
        jac[0, 3] = 2. * q0 * v1 + 2. * q1 * v2

        # dout1 / dq0
        jac[1, 0] = -2. * q3 * v0 + 4. * q0 * v1 + 2. * q1 * v2
        # dout1 / dq1
        jac[1, 1] = 2. * q2 * v0 + 2. * q0 * v2
        # dout1 / dq2
        jac[1, 2] = 2. * q1 * v0 + 4. * q2 * v1 + 2. * q3 * v2
        # dout1 / dq3
        jac[1, 3] = -2. * q0 * v0 + 2. * q2 * v2

        # dout2 / dq0
        jac[2, 0] = 2. * q2 * v0 - 2. * q1 * v1 + 4. * q0 * v2
        # dout1 / dq1
        jac[2, 1] = 2. * q3 * v0 - 2. * q0 * v1
        # dout1 / dq2
        jac[2, 2] = 2. * q0 * v0 + 2. * q3 * v1
        # dout1 / dq3
        jac[2, 3] = 2. * q1 * v0 + 2. * q2 * v1 + 4. * q3 * v2

        return jac

    def to(self, x, Type):
        if Type == SO3Space:
            return self.expand(x).reshape((-1, 1))
        elif Type == AxisAngleSpace:
            return log_quat(x)
        elif Type == NaiveQuaternionSpace or Type == UnitQuaternionSpace:
            return x
        assert(False and "Invalid space conversion!")

    @staticmethod
    def __H():
        return np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])

    @staticmethod
    def __L(*args):
        if len(args)==2:
            s=args[0]
            v=args[1]
            return np.array([
                [s, -v[0], -v[1], -v[2]],
                [v[0], s, -v[2], v[1]],
                [v[1], v[2], s, -v[0]],
                [v[2], -v[1], v[0], s]
            ])
        else:
            qq=args[0]
            q = qq.reshape((4, 1))
            s = q[0, 0]
            v = q[1:]

            Lm = np.zeros((4, 4))
            Lm[0, 0] = s
            Lm[1:, 0:1] = v
            Lm[0, 1:] = -v.T
            Lm[1:, 1:] = s*np.eye(3) + hat(v)
            return Lm

    @staticmethod
    def __R(*args):
        if len(args)==2:
            s=args[0]
            v=args[1]
            return np.array([
            [s, -v[0], -v[1], -v[2]],
            [v[0], s, v[2], -v[1]],
            [v[1], -v[2], s, v[0]],
            [v[2], v[1], -v[0], s]
            ])
        else:
            qq=args[0]
            q = qq.reshape((4, 1))
            s = q[0, 0]
            v = q[1:]

            Rm = np.zeros((4, 4))
            Rm[0, 0] = s
            Rm[1:, 0:1] = v
            Rm[0, 1:] = -v.T
            Rm[1:, 1:] = s*np.eye(3) - hat(v)

            return Rm

    @staticmethod
    def G(q):
        return UnitQuaternionSpace.__L(q) @ UnitQuaternionSpace.__H()

class NaiveQuaternionSpace(UnitQuaternionSpace):
    def __init__(self):
        super().__init__(False)

class CombinedSpace:
    def __init__(self, spaces):
        self._nq = 0
        self._nv = 0
        self._nd = 0

        for i in range(len(spaces)):
            self._nq += spaces[i]._nq
            self._nv += spaces[i]._nv
            self._nd += spaces[i].nd()

        self._spaces = copy.deepcopy(spaces)

    def nd(self):
        return self._nd

    def zero(self):
        nq = self._nq
        xn = np.zeros((nq, 1))
        q_idx = 0
        for i in range(len(self._spaces)):
            sp = self._spaces[i]

            xn[q_idx : q_idx + sp._nq] = sp.zero()

            q_idx += sp._nq
        return xn

    def expand(self, current):
        res = []
        q_idx = 0
        for i in range(len(self._spaces)):
            sp = self._spaces[i]
            x = current[q_idx : q_idx + sp._nq]

            res.append(sp.expand(x))
            q_idx += sp._nq
        return res

    def state_diff(self, target, current):
        diff = np.zeros((self._nd, 1))
        q_idx = 0
        d_idx = 0
        for i in range(len(self._spaces)):
            sp = self._spaces[i]
            tr = target[q_idx : q_idx + sp._nq]
            x = current[q_idx : q_idx + sp._nq]

            diff[d_idx : d_idx + sp.nd(), :] = sp.state_diff(tr, x)

            q_idx += sp._nq
            d_idx += sp.nd()
        return diff

    def state_diff_deriv(self, target, current, arg = 1):
        J = np.zeros((self._nd, self._nd))
        # v_idx = 0
        q_idx = 0
        d_idx = 0
        for i in range(len(self._spaces)):
            sp = self._spaces[i]
            tr = target[q_idx : q_idx + sp._nq]
            x = current[q_idx : q_idx + sp._nq]

            J[d_idx : d_idx + sp.nd(), d_idx : d_idx + sp.nd()] = sp.state_diff_deriv(tr, x, arg)

            q_idx += sp._nq
            # v_idx += sp._nv
            d_idx += sp.nd()
        return J

    def step(self, x, v, dt):
        nq = self._nq
        xn = np.zeros((nq, 1))
        v_idx = 0
        q_idx = 0
        for i in range(len(self._spaces)):
            sp = self._spaces[i]
            xc = x[q_idx : q_idx + sp._nq]
            vc = v[v_idx : v_idx + sp._nv]

            xn[q_idx : q_idx + sp._nq] = sp.step(xc, vc, dt)

            q_idx += sp._nq
            v_idx += sp._nv
        return xn

    def step_deriv(self, x, v, dt, arg = 0, so3 = False):
        nq = self._nq
        if so3:
            nq = self.nd()
        nv = self._nv
        if arg == 0:
            J = np.zeros((nq, nq))
            v_idx = 0
            q_idx = 0
            d_idx = 0
            for i in range(len(self._spaces)):
                sp = self._spaces[i]
                xc = x[q_idx : q_idx + sp._nq]
                vc = v[v_idx : v_idx + sp._nv]

                i_idx = q_idx
                i_idx_f = q_idx + sp._nq
                if so3:
                    i_idx = d_idx
                    i_idx_f = d_idx + sp.nd()

                J[i_idx : i_idx_f, q_idx : q_idx + sp._nq] = sp.step_deriv(xc, vc, dt, arg)
                q_idx += sp._nq
                v_idx += sp._nv
                d_idx += sp.nd()
            return J
        J = np.zeros((nq, nv))
        v_idx = 0
        q_idx = 0
        d_idx = 0
        for i in range(len(self._spaces)):
            sp = self._spaces[i]
            xc = x[q_idx : q_idx + sp._nq]
            vc = v[v_idx : v_idx + sp._nv]

            i_idx = q_idx
            i_idx_f = q_idx + sp._nq
            if so3:
                i_idx = d_idx
                i_idx_f = d_idx + sp.nd()

            J[i_idx : i_idx_f, v_idx : v_idx + sp._nv] = sp.step_deriv(xc, vc, dt, arg)
            q_idx += sp._nq
            v_idx += sp._nv
            d_idx += sp.nd()
        return J

    def to(self, current, types):
        if type(types) == CombinedSpace:
            types = [type(s) for s in types._spaces]
        if len(types) != len(self._spaces):
            assert("Invalid number of spaces")
        res = []
        q_idx = 0
        nq = 0
        for i in range(len(types)):
            sp = self._spaces[i]
            x = current[q_idx : q_idx + sp._nq]

            res.append(sp.to(x, types[i]))
            q_idx += sp._nq
            nq += res[-1].shape[0]
        q = np.zeros((nq, 1))
        q_idx = 0
        for i in range(len(res)):
            q[q_idx : q_idx + res[i].shape[0]] = res[i]
            q_idx += res[i].shape[0]
        return q
