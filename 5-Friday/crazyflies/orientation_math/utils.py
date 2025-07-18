import numpy as np

"""
Utility for pretty-printing states in different spaces.

Provides a function to split and expand a state vector according to a given space
(e.g., SO3Space, CombinedSpace, VectorSpace, etc.) and print each component nicely.
"""

def pretty_print_space(space, x):
    """
    Pretty-print the components of state vector x according to the provided space.

    Args:
        space: an instance of a state space (VectorSpace, SO3Space, CombinedSpace, etc.).
        x (np.ndarray): state vector of shape (nq, 1).
    """
    import numpy as np
    # Helper to format arrays
    def fmt(arr):
        arr = np.array(arr)
        with np.printoptions(precision=4, suppress=True):
            return np.array2string(arr.flatten(), separator=', ')

    # If CombinedSpace, iterate subspaces
    if hasattr(space, '_spaces'):
        idx = 0
        for sub in space._spaces:
            nq = sub._nq
            comp = x[idx:idx+nq]
            expanded = sub.expand(comp)
            name = type(sub).__name__
            print(f"--- {name} (dim={nq}) ---")
            print(expanded)
            idx += nq
    else:
        # Single space
        name = type(space).__name__
        print(f"--- {name} ---")
        try:
            expanded = space.expand(x)
            print(expanded)
        except Exception:
            # If no expand, print raw
            print(x)

# Example usage:
if __name__ == '__main__':
    import numpy as np
    from spaces import SO3Space, VectorSpace, CombinedSpace

    # SO(3) example
    so3 = SO3Space()
    x = so3.zero()  # identity
    print("SO3Space example:")
    pretty_print_space(so3, x)

    # VectorSpace example
    vs = VectorSpace(3)
    xv = vs.zero() + np.array([[1.0],[2.0],[3.0]])
    print("VectorSpace example:")
    pretty_print_space(vs, xv)

    # CombinedSpace example
    cs = CombinedSpace([vs, so3])
    xc = np.vstack([xv, x])
    print("CombinedSpace example:")
    pretty_print_space(cs, xc)
