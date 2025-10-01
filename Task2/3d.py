import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
A, B, C = 10.0, 28.0, 2.667


def lorenz(state):
    """
    Compute the derivatives for the Lorenz system.

    Parameters
    ----------
    state : ndarray
        Current state [x, y, z].

    Returns
    -------
    ndarray
        Derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dxdt = A * (y - x)
    dydt = B * x - y - x * z
    dzdt = x * y - C * z
    return np.array([dxdt, dydt, dzdt])


def rk4_step(func, state, dt):
    """
    Perform a single Runge-Kutta 4th order (RK4) step.

    Parameters
    ----------
    func : callable
        Function that computes derivatives.
    state : ndarray
        Current state [x, y, z].
    dt : float
        Time step.

    Returns
    -------
    ndarray
        Updated state after one step.
    """
    k1 = dt * func(state)
    k2 = dt * func(state + 0.5 * k1)
    k3 = dt * func(state + 0.5 * k2)
    k4 = dt * func(state + k3)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def main():
    """Simulate and plot the Lorenz system trajectory."""
    os.makedirs("outputs", exist_ok=True)
    # Simulation parameters
    dt = 0.01
    steps = 10_000
    state = np.array([0.0, 1.0, 1.05])  # Initial condition

    # Store trajectory
    trajectory = np.zeros((steps, 3))
    for i in range(steps):
        trajectory[i] = state
        state = rk4_step(lorenz, state, dt)

    # Extract x, y, z
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    # Plot 3D trajectory
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, lw=0.5, color="darkblue")
    ax.set_title("Bee's Path in 3D (Lorenz System with RK4)")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    plt.savefig("outputs/bee_path.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
