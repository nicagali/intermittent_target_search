# ...

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
    
# Define environment
env = TargetEnv(Nt=config['NUM_TARGETS'], L=config['WORLD_SIZE'], r=config['r'], rot_diff = config['ROT_DIFF'], trans_diff = config['TRANS_DIFF'], prop_vel=config['PROP_VEL'])


positions = [[env.positions[0][0]], [env.positions[0][1]]]

