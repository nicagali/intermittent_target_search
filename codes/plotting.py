import matplotlib.pyplot as plt
import local_mach_param as par
import plotting_param as plot_par
import numpy as np
from matplotlib.patches import Circle

def mask_array_pbc(positions, box_length):
    
    x_difference = np.abs(np.diff(positions[0]))    #size #positions-1
    y_difference = np.abs(np.diff(positions[1]))    #size #positions-1

    mask = (np.abs(x_difference) > box_length / 2) | (np.abs(y_difference) > box_length / 2)
    mask = np.hstack([mask, [False]]) #add one false so that size #positions
    x_masked_data = np.ma.MaskedArray(positions[0], mask)
    y_masked_data = np.ma.MaskedArray(positions[1], mask)
    
    return [x_masked_data, y_masked_data]
    

def plot_2d_trajectory(ax, positions, box_length, target, target_radius):
    
    positions = mask_array_pbc(positions, box_length)
    
    ax.plot(positions[0], positions[1], lw=1)
    ax.plot(positions[0][0], positions[1][0], 'go', label='Start')
    ax.plot(positions[0][-1], positions[1][-1], 'yo', label='End')
    ax.add_patch(Circle((target[0][0], target[0][1]), target_radius*box_length, fill=False, edgecolor='red'))

    # ax.plot(target[0][0], target[0][1], 'ro', markersize = , label='Target')
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    
    ax.set_aspect("equal")
    ax.legend()
    ax.grid()

def plot_msd(ax, msd, delta_t, env, activity):

    time = np.arange(0,len(msd))*delta_t

    ax.plot(time, msd, **plot_par.msd[f'{activity}_sim'], zorder=1)

    if activity == 'passive':
        msd_theory = 4*(env.trans_diff)*(time)
        ax.plot(time, msd_theory, **plot_par.msd[f'{activity}_theory'], zorder=0)
    if activity == 'active':
        msd_theory = ( 
                    4 * env.trans_diff * time +
                    ((2 * env.prop_vel**2) / env.rot_diff**2) * (env.rot_diff * time - 1 + np.exp(-env.rot_diff * time))
                    )
        ax.plot(time, msd_theory, **plot_par.msd[f'{activity}_theory'], zorder=0)

    ax.set_xlabel('t', fontsize = plot_par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=plot_par.size_ticks)
    ax.set_ylabel(r'MSD$\langle (r(t)-r_0)^2 \rangle$', fontsize = plot_par.axis_fontsize)
    ax.grid(':')
    ax.legend(fontsize = plot_par.legend_size)
