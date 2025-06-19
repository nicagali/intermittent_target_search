import matplotlib.pyplot as plt
import local_mach_param as par

def plot_2d_trajectory(ax, positions):
    ax.plot(positions[0], positions[1], lw=1)
    ax.plot(positions[0][0], positions[1][0], 'go', label='Start')
    ax.plot(positions[0][-1], positions[1][-1], 'ro', label='End')
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    
    ax.set_aspect("equal")
    ax.legend()
    ax.grid()