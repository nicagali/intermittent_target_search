import yaml
from rl_opts.learn_and_bench import *

CONFIGURATIONS_PATH = '/Users/monicaconte/PhD/Projects/Active_Matter/intermittent_target_search/data/configurations/'
RESULTS_PATH = '/Users/monicaconte/PhD/Projects/Active_Matter/intermittent_target_search/data/results/'

# Read configuration file
with open(f'{CONFIGURATIONS_PATH}exp0.cfg') as f:
    config = yaml.safe_load(f)
    
# learning(config, RESULTS_PATH, run = 1)


