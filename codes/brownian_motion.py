import numpy as np
import matplotlib.pyplot as plt

def active_brownian_motion(T=int(1/0.001), dt=0.001, v=1.0, D=0.1, D_theta=0.05, phi = 0, seed=None):
    """
    Simulates active Brownian motion in 2D.
    
    Parameters
    ----------
    T : float
        Total simulation time.
    dt : float
        Time step.
    v : float
        Self-propulsion speed.
    D : float
        Translational diffusion coefficient.
    D_theta : float
        Rotational diffusion coefficient.
    seed : int or None
        Random seed for reproducibility.
    
    Returns
    -------
    x, y : arrays
        Trajectory coordinates.
    """
    
    if seed is not None:
        np.random.seed(seed)
        
    N = T
    x = np.zeros(N)
    y = np.zeros(N)
    theta = np.zeros(N)
    
    # Initial orientation
    theta[0] = np.random.uniform(0, 2*np.pi)
    phi=1
    for t in range(1, N):
        # Unit vector
        u = np.array([np.cos(theta[t-1]), np.sin(theta[t-1])])
        
        # Random noise
        xi = np.random.normal(0, 1, size=2)
        eta = np.random.normal(0, 1)
        
        # Position update
        x[t] = x[t-1] + v * u[0] * phi * dt + np.sqrt(2 * D * dt) * xi[0]
        y[t] = y[t-1] + v * u[1] * phi * dt + np.sqrt(2 * D * dt) * xi[1]
        
        # Orientation update
        theta[t] = theta[t-1] + np.sqrt(2 * D_theta * dt) * eta
    
    return x, y

# Example usage
x, y = active_brownian_motion(T=int(1/0.001), dt=0.001, v=1.0, D=0.1, D_theta=0.05, seed=40)

# Plot trajectory
plt.figure(figsize=(6, 6))
plt.plot(x, y, lw=1)
plt.plot(x[0], y[0], 'go', label='Start')
plt.plot(x[-1], y[-1], 'ro', label='End')
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.savefig("motion.pdf")
