import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy

matplotlib.use("TkAgg")  # or 'TkAgg'

from quadrotor_env import QuadrotorEnv
from spaces import SO3Space
from math_utils import exp_rotation
from utils import pretty_print_space

orientation = SO3Space()
quad = QuadrotorEnv(orientation, tf = 100., dt = 0.01)

# QuadrotorEnv has the following key functions/properties
### Functions
# env.A(x_ref, u_ref): computes the partial derivatives of a forward integration step wrt to the state (x), around (x_ref, u_ref)
# env.B(x_ref, u_ref): computes the partial derivatives of a forward integration step wrt to the control (u), around (x_ref, u_ref)
# env._full_space.state_diff(x1, x2): computes the correct difference x1 - x2 (when in SO(3) for example, we cannot just use simple -)
### Properties
# env._m: the mass of the quadrotor
# env._g: the gravity acceleration (value -> positive)

# Stable point
u_ref = np.ones((4, 1)) * (quad._m * quad._g / 4.)
p0 = np.array([[0., 0., 0.]]).T
q0 = np.array([[1., 0., 0., 0.]]).T
r0 = np.eye(3).reshape((-1, 1))
v0 = np.zeros((3, 1))
omega0 = np.zeros((3, 1))

x_ref = np.concatenate([p0, r0, v0, omega0], axis=0)

# Linearize over (x_ref, u_ref)
A = ### TO-DO: Get the partial derivatives wrt the state (x)
B = ### TO-DO: Get the partial derivatives wrt the control (u)

N_lqr = quad._full_space._nv
M = quad._nu
K = quad._K

# The linearized system is controllable!
print(A.shape)
print(B.shape)
print(np.linalg.matrix_rank(A))

### LQR ###
# Cost
Q = ### TO-DO: Fill this. It should be a positive semi-definite matrix of size N_lqr X N_lqr
Rw = ### TO-DO: Fill this. It should be a positive definite matrix of size M X M
QN = ### TO-DO: Fill this. It should be a positive semi-definite matrix of size N_lqr X N_lqr

# Init
Ps = [np.zeros((N_lqr, N_lqr))]*K
Ks = [np.zeros((M, N_lqr))]*(K-1)

# Riccati backward
Ps[K-1] = QN
for k in range(K-2, -1, -1):
    ### TO-DO: Implement the Riccati recursion. Fill the values of the list Ps, Ks
    Ks[k] = # fill the K matrix for time k
    Ps[k] = # fill the P matrix for time k
### END OF LQR ###

# We approximate infinite horizon LQR with many many steps, and wet get the Kinf as one of the converged matrices
Kinf = Ks[0]

def save_Kinf_mat(Kinf, file):
    lines = []
    for row in Kinf:
        formatted_row = ", ".join(f"{val:.8e}f" for val in row)
        lines.append(f"{{{formatted_row}}}")
    cpp_struct_str = "static const float Kinf[4][12] = {\n"
    cpp_struct_str += ",\n".join(f"    {line}" for line in lines)
    cpp_struct_str += "};\n"
    with open(file, "w") as f:
        f.write(cpp_struct_str)
    print(f"Kinf matrix saved to \"{file}\" in C-style struct format.")

save_Kinf_mat(Kinf, "Kinf_thrusts.txt")

# exit(1)

def policy(x, x_ref):
    # Compute delta
    dx = quad._full_space.state_diff(x, x_ref)

    return ### TO-DO: Compute feedback policy. Remember that we are linearizing around u_ref

# Random Initial State
p = p0 + np.random.uniform(low=-2., high=2., size=(3, 1))

r = (r0.reshape((3, 3)) @ exp_rotation(np.random.uniform(low=-1., high=1., size=(3, 1)))).reshape((-1, 1))

x = np.concatenate([p, r, v0, omega0], axis=0)

# We can also change the target
p_t = np.array([[0., 0., 1.]]).T
r_t = r0.copy()
v_t = np.zeros((3, 1))
omega_t = np.zeros((3, 1))

x_target = np.concatenate([p_t, r_t, v_t, omega_t], axis=0)

np.set_printoptions(linewidth=300)

# Visualization
plt.ion()
fig = plt.figure(figsize=(10, 10))
# for stopping simulation with the esc key.
fig.canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])

ax = fig.add_subplot(111, projection='3d')

l = quad._l
m = quad._m
g = quad._g

p1 = np.array([[10. * l, 0., 0.]]).T
p2 = np.array([[-10. * l, 0., 0.]]).T
p3 = np.array([[0., 10. * l, 0.]]).T
p4 = np.array([[0., -10. * l, 0.]]).T

last_u = np.zeros((M, 1))

for k in range(K):    
    ### Plot Animation ##
    p, rot, v, omega = quad._full_space.expand(x)

    p1_t = rot @ p1 + x[:3]
    p2_t = rot @ p2 + x[:3]
    p3_t = rot @ p3 + x[:3]
    p4_t = rot @ p4 + x[:3]

    plt.cla()

    # Plot Quadrotor
    ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]], [p1_t[1], p2_t[1], p3_t[1], p4_t[1]], [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.', markersize=10., zorder=6)
    ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]], [p1_t[2], p2_t[2]], 'r-', zorder=4)
    ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]], [p3_t[2], p4_t[2]], 'r-', zorder=4)

    # Plot controls
    p1_e = rot @ (p1 + np.array([[0., 0., 0.2 * last_u[0, 0] / (0.25*m*g)]]).T) + x[:3]
    p2_e = rot @ (p2 + np.array([[0., 0., 0.2 * last_u[2, 0] / (0.25*m*g)]]).T) + x[:3]
    p3_e = rot @ (p3 + np.array([[0., 0., 0.2 * last_u[1, 0] / (0.25*m*g)]]).T) + x[:3]
    p4_e = rot @ (p4 + np.array([[0., 0., 0.2 * last_u[3, 0] / (0.25*m*g)]]).T) + x[:3]

    ax.plot([p1_t[0], p1_e[0]], [p1_t[1], p1_e[1]], [p1_t[2], p1_e[2]], 'b-', zorder=4)
    ax.plot([p2_t[0], p2_e[0]], [p2_t[1], p2_e[1]], [p2_t[2], p2_e[2]], 'b-', zorder=4)
    ax.plot([p3_t[0], p3_e[0]], [p3_t[1], p3_e[1]], [p3_t[2], p3_e[2]], 'b-', zorder=4)
    ax.plot([p4_t[0], p4_e[0]], [p4_t[1], p4_e[1]], [p4_t[2], p4_e[2]], 'b-', zorder=4)

    # Plot extra side
    p1_t = rot @ (p1 + np.array([[0., 0., -0.01]]).T) + x[:3]
    p2_t = rot @ (p2 + np.array([[0., 0., -0.01]]).T) + x[:3]
    p3_t = rot @ (p3 + np.array([[0., 0., -0.01]]).T) + x[:3]
    p4_t = rot @ (p4 + np.array([[0., 0., -0.01]]).T) + x[:3]
    ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]], [p1_t[1], p2_t[1], p3_t[1], p4_t[1]], [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'g.', markersize=14., zorder=4)

    plt.xlim(-2., 2.)
    plt.ylim(-2., 2.)
    ax.set_zlim(0., 4.)

    plt.pause(0.0001)

    ### Control
    # u_new = policy(x, x_target) # Unbounded control; we allow LQR to cheat!
    u_new = np.maximum(policy(x, x_target), 0.) # The motors cannot generate thrust in the opposite direction; LQR does not knows this, but we need to clamp it! No cheating!
    ### Simulation step
    x = quad.step(x, u_new)

    last_u = np.copy(u_new)

plt.close()
