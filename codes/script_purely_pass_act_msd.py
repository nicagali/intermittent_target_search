# Motion with fixed only passive (old_phase=0, new_phase=0) or only active motion (old_phase=1, new_phase=1)

import yaml
import rl_opts
from rl_opts.rl_framework.legacy import TargetEnv, Forager
import matplotlib.pyplot as plt
import local_mach_param as par
import plotting
import numpy as np

# Read configuration file
with open(f'{par.CONFIGURATIONS_PATH}exp1.cfg') as f:
    config = yaml.safe_load(f)

def compute_msd(env, activity, timesteps, delta_t):
    """
    Computes msd of a fixed active or passive policy. No boundary conditions.

    Parameters
    ----------
    env : class
    Initial configuration of num_agents, parameters of active and passive diffusion.

    Returns
    -------
    Mean squared displacement.

    """

    msd = []

    initial_positions = np.copy(env.positions)

    for _ in range(1, timesteps):

        old_phase = 0
        new_phase = 0
        if activity == 'active':
            old_phase = 1
            new_phase = 1

        for agent_index in range(env.num_agents):
            env.update_pos(old_phase=old_phase, new_phase=new_phase, delta_t = delta_t, agent_index=agent_index)  

        displacements = env.positions - initial_positions  # shape (N, 2)
        squared_displacements = np.sum(displacements**2, axis=1)  # sum over x and y for each particle, shape (N,)
        msd_time_step = np.mean(squared_displacements)
        msd.append(msd_time_step)

    return msd

# Parameters

TIME_STEPS = config['NUM_TIME_STEPS'] #
NUM_AGENTS = 1000
    
# Define environment
env = TargetEnv(rot_diff = config['ROT_DIFF'], trans_diff = config['TRANS_DIFF'], prop_vel=config['PROP_VEL'], num_agents = NUM_AGENTS)

# Compute msd in passive motion
delta_t = 0.001
activity = 'passive'
# msd = compute_msd(env, activity=activity, timesteps=TIME_STEPS, delta_t=delta_t)
# np.save(par.RESULTS_PATH+'msd'+activity+'.npy', msd)
msd = np.load(par.RESULTS_PATH + 'msd' + activity + '.npy')

fig, ax = plt.subplots()
plotting.plot_msd(ax, msd, delta_t, env, activity=activity)
fig.tight_layout()
plt.savefig(f"{par.PLOT_PATH}msd_passive.pdf")

# Define environment
env = TargetEnv(rot_diff = config['ROT_DIFF'], trans_diff = config['TRANS_DIFF'], prop_vel=config['PROP_VEL'], num_agents = NUM_AGENTS)

# Compute msd in active motion
delta_t = 0.001
activity = 'active'
# msd = compute_msd(env, activity=activity, timesteps=TIME_STEPS, delta_t=delta_t)
# np.save(par.RESULTS_PATH+'msd'+activity+'.npy', msd)
msd = np.load(par.RESULTS_PATH + 'msd' + activity + '.npy')

fig, ax = plt.subplots()
plotting.plot_msd(ax, msd, delta_t, env, activity=activity)
fig.tight_layout()
plt.savefig(f"{par.PLOT_PATH}msd_active.pdf")
