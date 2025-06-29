# Motion with fixed only passive (old_phase=0, new_phase=0) or only active motion (old_phase=1, new_phase=1)

import yaml
import rl_opts
from rl_opts.rl_framework.legacy import TargetEnv, Forager, PSAgent
from rl_opts.learn_and_walk import learning
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

            
learning(config, par.RESULTS_PATH, num_agents=10)
    

