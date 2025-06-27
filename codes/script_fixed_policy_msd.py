# With fixed h matrix to not changoing position [1,0] calculate msd in a space without bc over one tau. Loop over timesteps in tau and over N agent classes.

import yaml
import rl_opts
from rl_opts.rl_framework.legacy import TargetEnv, Forager, PSAgent
import matplotlib.pyplot as plt
import local_mach_param as par
import plotting
import numpy as np

# Read configuration file
with open(f'{par.CONFIGURATIONS_PATH}exp1.cfg') as f:
    config = yaml.safe_load(f)
    
# Check insterted parameters
if config['NUM_TIME_STEPS'] < config['NUM_BINS']:
    raise ValueError(f"Parameters do not match: more bins than timesteps")
    
def compute_msd(env, agents, timesteps, delta_t):
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

        for agent_index, agent in enumerate(agents):

            old_phase = int(agent.phase)
            phase_duration = int(agent.bin_state())
            # print(old_phase, agent.duration, phase_duration)
            
            observation = [old_phase, phase_duration] # [phi, w]
            # print(observation)
            
            action  = agent.deliberate(observation)
            # print(action)
            
            agent.act(action)
            # print(agent.phase, agent.duration)
            
            env.update_pos(old_phase=old_phase, new_phase=agent.phase, delta_t = delta_t, agent_index = agent_index)  

        displacements = env.positions - initial_positions  # shape (N, 2)
        squared_displacements = np.sum(displacements**2, axis=1)  # sum over x and y for each particle, shape (N,)
        msd_time_step = np.mean(squared_displacements)
        msd.append(msd_time_step)

    return msd



# Parameters  
TAU = int(config['WORLD_SIZE']**2 / (4 * config['TRANS_DIFF'])) #characteristic passive time
TIME_STEPS_TAU = config['NUM_TIME_STEPS'] #
delta_t = TAU / TIME_STEPS_TAU
NUM_TAU = config['NUM_TAUS'] #number of times tau per episode
EPISODES = config['NUM_EPISODES'] #number of episodes
    
NUM_AGENTS = 1000
    
# Define environment
env = TargetEnv(rot_diff = config['ROT_DIFF'], trans_diff = config['TRANS_DIFF'], prop_vel=config['PROP_VEL'], num_agents = NUM_AGENTS)

#initialize agent 
NUM_PERCEPT_LIST = [
    2,  # phi: passive or active
    config['NUM_BINS']]  # w: binned duration in phase
STATE_SPACE = [np.arange(2), np.linspace(0, TIME_STEPS_TAU-1, config['NUM_BINS'])]
NUM_STATES = np.prod([len(i) for i in STATE_SPACE])

#default initialization policy
INITIAL_DISTR = []
for percept in range(NUM_STATES):
    # Intialize as passive
    INITIAL_DISTR.append([1, 0]) # in a state, [prob of not change, prob of change]

# print(STATE_SPACE, NUM_STATES)    

agents = [Forager(state_space = STATE_SPACE,
            num_actions=config['NUM_ACTIONS'],
            num_percepts_list=NUM_PERCEPT_LIST,
            gamma_damping=config['GAMMA'],
            eta_glow_damping=config['ETA_GLOW'],
            initial_prob_distr=INITIAL_DISTR) for _ in range(env.num_agents)]


# Compute msd in passive motion
delta_t = 0.001
activity = 'passive'
# msd = compute_msd(env, agents, activity=activity, timesteps=TIME_STEPS_TAU, delta_t=delta_t)
# np.save(par.RESULTS_PATH+'msd'+activity+'.npy', msd)
msd = np.load(par.RESULTS_PATH + 'msd' + activity + '.npy')

fig, ax = plt.subplots()
plotting.plot_msd(ax, msd, delta_t, env)
fig.tight_layout()
plt.savefig(f"{par.PLOT_PATH}msd_passive_fixed_pol.pdf")

# Compute msd in active motion
delta_t = 0.001
activity = 'active'
msd = compute_msd(env, agents, timesteps=TIME_STEPS_TAU, delta_t=delta_t)
np.save(par.RESULTS_PATH+'msd'+activity+'pol.npy', msd)
# msd = np.load(par.RESULTS_PATH + 'msd' + activity + 'pol.npy')

fig, ax = plt.subplots()
plotting.plot_msd(ax, msd, delta_t, env, activity=activity)
fig.tight_layout()
plt.savefig(f"{par.PLOT_PATH}msd_active_fixed_pol.pdf")
