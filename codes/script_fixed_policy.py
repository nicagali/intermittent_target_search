# Motion with fixed only passive (old_phase=0, new_phase=0) or only active motion (old_phase=1, new_phase=1)

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
    
# Parameters  
TAU = int(config['WORLD_SIZE']**2 / (4 * config['TRANS_DIFF'])) #characteristic passive time
TIME_STEPS_TAU = config['NUM_TIME_STEPS'] #
delta_t = TAU / TIME_STEPS_TAU
NUM_TAU = config['NUM_TAUS'] #number of times tau per episode
EPISODES = config['NUM_EPISODES'] #number of episodes
    
# print('tau', TAU, 'num_time_steps', config['NUM_TIME_STEPS'],  'delta t', DELTA_T)
    
# Define environment
env = TargetEnv(Nt=config['NUM_TARGETS'], L=config['WORLD_SIZE'], r=config['r'], rot_diff = config['ROT_DIFF'], trans_diff = config['TRANS_DIFF'], prop_vel=config['PROP_VEL'])
positions = [[env.positions[0][0]], [env.positions[0][1]]]

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

print(STATE_SPACE, NUM_STATES)    

agent = Forager(state_space = STATE_SPACE,
                num_actions=config['NUM_ACTIONS'],
                num_percepts_list=NUM_PERCEPT_LIST,
                gamma_damping=config['GAMMA'],
                eta_glow_damping=config['ETA_GLOW'],
                initial_prob_distr=INITIAL_DISTR)

# print(agent.num_percepts)

# observation = [1,0] # [phi, w]

# Loop over time steps of one tau

for time_index in range(1, TIME_STEPS_TAU+1):
    
    time = time_index*delta_t
    print('time_index', time_index)
    
    old_phase = int(agent.phase)
    phase_duration = int(agent.bin_state())
    print(old_phase, agent.duration, phase_duration)
    
    observation = [old_phase, phase_duration] # [phi, w]
    print(observation)
    
    state = agent.percept_preprocess(observation)
    print(state, agent.percept_preprocess([0,1]))
    
    action  = agent.deliberate(observation)
    print(action)
    
    agent.act(action)
    print(agent.phase, agent.duration)
    
    env.update_pos(old_phase=old_phase, new_phase=agent.phase, delta_t = delta_t)  
    
    found_target_pos = np.copy(env.target_positions) 
    
    reward = env.check_encounter()
        
    env.check_bc()
    
    positions[0].append(env.positions[0][0])
    positions[1].append(env.positions[0][1])
#     # if reward==1:
#     #     print(reward)
#     #     break
    
fig, ax = plt.subplots()
plotting.plot_2d_trajectory(ax, positions, env.L, found_target_pos, env.r)
plt.savefig(f"{par.PLOT_PATH}motion_class_agent.pdf")

