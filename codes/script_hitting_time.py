# Motion with fixed only passive (old_phase=0, new_phase=0) or only active motion (old_phase=1, new_phase=1)

import yaml
import rl_opts
from rl_opts.rl_framework.legacy import TargetEnv, Forager, PSAgent
from rl_opts.learn_and_walk import avg_hitting_time
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
    
EPISODES = 10
NUM_AGENTS = 1000

avg_hitting_time_vec = []

for episode_index in range(EPISODES):
    print(episode_index)

    policy = np.load(par.RESULTS_PATH+'memory_agent_episode_'+str(episode_index+1)+'.npy')    
    # print(episode_index, policy)

    avg_hitting_time_value = avg_hitting_time(config, result_path=par.RESULTS_PATH, num_agents=NUM_AGENTS,  policy=policy)
    avg_hitting_time_vec.append(avg_hitting_time_value)
    
np.save(par.RESULTS_PATH+'avg_hitting_time.npy', avg_hitting_time)

fig, ax = plt.subplots()
plotting.plot_avg_hitting_time(ax, avg_hitting_time_vec)
plt.savefig(f"{par.PLOT_PATH}avg_hitting_time.pdf")


