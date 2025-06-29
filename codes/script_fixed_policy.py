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
    
NUM_AGENTS = 100
    
# Define environment
env = TargetEnv(L=config['WORLD_SIZE'], r=config['r'], rot_diff = config['ROT_DIFF'], trans_diff = config['TRANS_DIFF'], prop_vel=config['PROP_VEL'], num_agents=NUM_AGENTS)
positions = [[env.positions[0][0]], [env.positions[0][1]]]

#initialize agent s
NUM_PERCEPT_LIST = [
    2,  # phi: passive or active
    config['NUM_BINS']]  # w: binned duration in phase
STATE_SPACE = [np.arange(2), np.linspace(0, TIME_STEPS_TAU-1, config['NUM_BINS'])]
NUM_STATES = np.prod([len(i) for i in STATE_SPACE])

#default initialization policy per agent
INITIAL_DISTR = []
for percept in range(NUM_STATES):
    # Intialize as passive
    INITIAL_DISTR.append([1, 0]) # in a state, [prob of not change, prob of change]
policy = np.load(par.RESULTS_PATH+'memory_agent_'+str(1)+'_episode_'+str(50)+'.npy')
# print(STATE_SPACE, NUM_STATES)    

agents = [Forager(state_space = STATE_SPACE,
        num_actions=config['NUM_ACTIONS'],
        num_percepts_list=NUM_PERCEPT_LIST,
        gamma_damping=config['GAMMA'],
        eta_glow_damping=config['ETA_GLOW'],
        initial_prob_distr=INITIAL_DISTR) for _ in range(NUM_AGENTS)]

# Loop over time steps of one tau
Done = False
hitting_time = 0
for agent_index, agent in enumerate(agents):
    
    agent.h_matrix = policy

    for time_index in range(1, (NUM_TAU*TIME_STEPS_TAU+1)):
        # print('index', time_index)
        old_phase = int(agent.phase)
        phase_duration = int(agent.bin_state())
        # print(old_phase, agent.duration, phase_duration)
        
        observation = [old_phase, phase_duration] # [phi, w]
        # print(observation)
        
        action  = agent.deliberate(observation)
        # print(action)
        
        agent.act(action)
        # print(agent.phase, agent.duration)
        
        env.update_pos(old_phase=old_phase, new_phase=agent.phase, delta_t = delta_t, agent_index=agent_index)  
        
        if agent_index==0:
            found_target_pos = np.copy(env.target_positions) 
        
        reward = env.check_encounter_full_target(agent_index=agent_index)
            
        env.check_bc(agent_index)
        
        if agent_index==0: #save positions of agent 0
            positions[0].append(env.positions[0][0])
            positions[1].append(env.positions[0][1])
        if reward==1:
            print('agent', agent_index+1, 'reward', reward, 'at time', time_index)
            hitting_time += time_index
            break

    hitting_time += time_index
        
avg_hitting_time_over_tau = hitting_time/(TIME_STEPS_TAU*NUM_AGENTS)
print(hitting_time, avg_hitting_time_over_tau)
print(f'First hitting time = {avg_hitting_time_over_tau}tau')

    
fig, ax = plt.subplots()
plotting.plot_2d_trajectory(ax, positions, env.L, found_target_pos, env.r)
plt.savefig(f"{par.PLOT_PATH}motion_class_agent.pdf")

