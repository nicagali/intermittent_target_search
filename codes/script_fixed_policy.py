# Motion with fixed only passive (old_phase=0, new_phase=0) or only active motion (old_phase=1, new_phase=1)

import yaml
import rl_opts
from rl_opts.rl_framework.legacy import TargetEnv, Forager, PSAgent
from rl_opts.learn_and_walk import walk_from_policy
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
    
EPISODE = 1
policy = np.load(par.RESULTS_PATH+'memory_agent_episode_'+str(EPISODE)+'.npy')    

walk_from_policy(config, result_path=par.RESULTS_PATH, e=EPISODE,  policy=policy)

positions = np.load(par.RESULTS_PATH+'path_agent_episode_'+str(EPISODE)+'.npz')
print(positions['positions'])
    
fig, ax = plt.subplots()
plotting.plot_2d_trajectory(ax, positions['positions'], config['WORLD_SIZE'], positions['found_target_positions'], config['r'])
plt.savefig(f"{par.PLOT_PATH}motion_class_agent.pdf")

