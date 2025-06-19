import yaml
import rl_opts
from rl_opts.rl_framework.legacy import TargetEnv, Forager
import matplotlib.pyplot as plt
import local_mach_param as par
import plotting
import numpy as np

CONFIGURATIONS_PATH = '/Users/monicaconte/PhD/Projects/Active_Matter/intermittent_target_search/data/configurations/'
RESULTS_PATH = '/Users/monicaconte/PhD/Projects/Active_Matter/intermittent_target_search/data/results/'

# Read configuration file
with open(f'{CONFIGURATIONS_PATH}exp0.cfg') as f:
    config = yaml.safe_load(f)
    
# Define environment

env = TargetEnv(Nt=config['NUM_TARGETS'], L=config['WORLD_SIZE'], r=config['r'], rot_diff = config['ROT_DIFF'], trans_diff = config['TRANS_DIFF'], prop_vel=config['PROP_VEL'])
# print(env.positions)

positions = [[env.positions[0][0]], [env.positions[0][1]]]
print(config['DELTA_T'], int(1/config['DELTA_T']))

for time in range(int(1/config['DELTA_T'])):
    
    env.update_pos(old_phase=1, new_phase=1, delta_t = config['DELTA_T'])    
    positions[0].append(env.positions[0][0])
    positions[1].append(env.positions[0][1])
    
    
    
fig, ax = plt.subplots()
plotting.plot_2d_trajectory(ax, positions)
plt.savefig(f"{par.PLOT_PATH}motion_class.pdf")
    