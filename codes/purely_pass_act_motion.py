import yaml
import rl_opts
from rl_opts.rl_framework.legacy import TargetEnv, Forager
import matplotlib.pyplot as plt
import local_mach_param as par
import plotting
import numpy as np

# np.random.seed(40)
# st0 = np.random.get_state()
# print(st0)

# Read configuration file
with open(f'{par.CONFIGURATIONS_PATH}exp0.cfg') as f:
    config = yaml.safe_load(f)
    
# Define environment
env = TargetEnv(Nt=config['NUM_TARGETS'], L=config['WORLD_SIZE'], r=config['r'], rot_diff = config['ROT_DIFF'], trans_diff = config['TRANS_DIFF'], prop_vel=config['PROP_VEL'])

# Loop over
positions = [[env.positions[0][0]], [env.positions[0][1]]]
for time in range(int(1/config['DELTA_T'])):
    
    env.update_pos(old_phase=0, new_phase=0, delta_t = config['DELTA_T'])  
    
    found_target_pos = np.copy(env.target_positions)
    
    reward = env.check_encounter()
        
    env.check_bc()
    
    positions[0].append(env.positions[0][0])
    positions[1].append(env.positions[0][1])
    if reward==1:
        print(reward)
        break
    
fig, ax = plt.subplots()
plotting.plot_2d_trajectory(ax, positions, env.L, found_target_pos, env.r)
plt.savefig(f"{par.PLOT_PATH}motion_class.pdf")
    