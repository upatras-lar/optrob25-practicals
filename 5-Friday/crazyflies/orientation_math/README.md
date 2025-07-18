# Orientation Math

Reference implementation of common orientation math (Lie Algebra, Axis-Angle, Quaternions, Quat-Trick, Euler Angles). We provide:

* **Rotation utilities** (`math_utils.py`):

  * Angle normalization, conversion between degrees/radians
  * Exponential/log maps for SO(3), Jacobians and inverses
  * Conversions between rotation matrices, Euler angles, quaternions, axis-angle
  * Helper for computing Jacobians of cross products

* **State spaces** (`spaces.py`):

  * `VectorSpace`, `SO3Space`, `AxisAngleSpace`, `EulerAnglesSpace`, `UnitQuaternionSpace`, `NaiveQuaternionSpace`, `CombinedSpace`
  * Consistent interfaces for `expand`, `state_diff`, `step`, `step_deriv`, `to` conversions
  * List of available spaces:
    * `VectorSpace`: simple vector space.
    * `SO3Space`: Represent rotations as a matrix Lie Group (SO(3)): see [https://arxiv.org/abs/1812.01537](https://arxiv.org/abs/1812.01537) for details.
    * `AxisAngleSpace`: Represent rotations in the tangent space of SO(3): see [https://arxiv.org/abs/1812.01537](https://arxiv.org/abs/1812.01537) for details.
    * `UnitQuaternionSpace`: Represent rotations as quaternions and utilize the ``Quat-Trick'': see [https://rexlab.ri.cmu.edu/papers/planning_with_attitude.pdf](https://rexlab.ri.cmu.edu/papers/planning_with_attitude.pdf) for details.
    * `NaiveQuaternionSpace`: Represent rotations as unit-norm quaternions.
    * `CombinedSpace`: class to represent hybrid spaces (e.g. translation + rotation).

* **Quadrotor environment** (`quadrotor_env.py`):

  * `QuadrotorEnv` simulating a 6-DOF quadrotor with arbitrary orientation space
  * Semi-implicit Euler integration, analytic Jacobians `A` and `B`
  * Serving mostly as an example on how to use the rest of the code

* **Examples and Utilities**:

  * `simple_examples.py`, `spaces_examples.py`, `quadrotor_env_examples.py`: runnable examples
  * `utils.py`: `pretty_print_space` for human-readable state inspection

## Disclaimer

This package is intended primarily as a **reference implementation** to gather together common orientation/math utilities.  
It is **not** highly optimized for performance, nor guaranteed to handle every edge case—use it at your own risk and verify results thoroughly in critical applications.

## Gradient Tests

A validation script using finite differences to verify analytic Jacobians and derivatives for state spaces and environments. To run:

```bash
python tests/gradient_tests.py
```

## Package Structure

```
orientation_math/
├── math_utils.py
├── spaces.py
├── utils.py
├── quadrotor_env.py
├── examples
├──── rotation_examples.py
├──── state_spaces_examples.py
├──── quadrotor_env_examples.py
├── tests
├──── gradient_tests.py
└── README.md
```

## Quick Start

```python
from math_utils import angle_dist, deg2rad, hat, exp_rotation, log_rotation, quaternion_to_rotation_matrix
from spaces import SO3Space, VectorSpace, CombinedSpace
from utils import pretty_print_space

# Normalize an angle
theta = angle_dist(b=3.5, a=0.1)
# Rotate a vector
R = exp_rotation(np.array([[0.2],[0.1],[0.]]))
# Convert quaternion to rotation matrix
q = np.array([[0.9830],[0.1830],[0.0450],[0.}}])
R_q = quaternion_to_rotation_matrix(q)

# Use state space
space = CombinedSpace([VectorSpace(3), SO3Space()])
x = space.zero()
pretty_print_space(space, x)
```

## Quadrotor Simulation

```python
from spaces import SO3Space
from quadrotor_env import QuadrotorEnv

env = QuadrotorEnv(SO3Space(), cost=None)
x0 = eenv._space.zero()  # hover state
u0 = np.array([[m*g/4]]*4)
x1 = env.step(x0, u0)
```

## License

MIT License
