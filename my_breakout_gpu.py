
"""
This code borrows heavily from rhiga2's OpenAI gym implementation of Cartpole,
at https://github.com/rhiga2/cartpole_dqn
"""


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple
import time
import scipy.misc

from collections import OrderedDict
import cv2
import os

np.set_printoptions(precision=4)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'nonterminal'))

STATE_DEPTH = 4
IMG_DEPTH = 1
FRAME_SHAPE = (IMG_DEPTH, 84, 84)
INPUT_SHAPE = (STATE_DEPTH, 84, 84)
FILTERS_1 = 32
FILTERS_2 = 64
FILTERS_3 = 64 


class episode():
    def __init__(self):
        self.state = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        self.action = -1
        self.next_state = None
        self.reward = -1
        self.nonterminal = False
######################################################################
# Replay Memory object has no attribute 'save_state_dict'
# Stores state, action, next_state, reward, done

class ReplayMemory():
    def __init__(self, capacity = 1000000):
        ''' Initializes empty replay memory '''
        self.capacity = capacity
        self.memory = [episode() for i in range(capacity)]
        self.position = 0
        self.real_capacity = 0
        self.done_indices = []
        self.sensitive_indices = []

    def push(self, item):
        ''' Stores item in replay memory '''
        """
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
            self.position = (self.position + 1) % self.capacity
        """
        if (self.position+1 in self.done_indices):
            self.done_indices.remove(self.position+1)
        x = self.memory[self.position]
        x.state[:,:,:] = item[0][:,:,:]
        x.action = item[1]
        x.reward = item[3]
        x.nonterminal = item[4]
        self.position = (self.position+1) % self.capacity
        if (self.real_capacity < self.capacity):
            self.real_capacity += 1

    def sample(self, batch_size):
        ''' Samples item from replay memory '''
        #if (random.random() > 0.8 and len(self.sensitive_indices) > batch_size):
        #    indices = random.sample(self.sensitive_indices, batch_size)
        #else:
        terminal_num = 0
        #indices = random.sample(self.done_indices, terminal_num)
        indices = random.sample(range(self.real_capacity-1), batch_size - terminal_num) #+ indices
        indices = [i for i in indices if self.index_valid(i)]
        samples = np.array([(self.memory[i].state, self.memory[i].action, None, self.memory[i].reward, self.memory[i].nonterminal) for i in indices])
       
        
        return samples, indices
        
    def next_states(self, indices):
        next_states = np.array([self.memory[i+1].state for i in indices])
        return next_states
    
    def purge(self):
        if (len(self.memory) > 20000):
            amount = self.done_indices[5]
            self.memory = np.delete(np.array(self.memory), range(amount)).list()
            self.states = np.delete(self.states, range(amount))
            self.done_indices = np.delete(np.array(self.done_indices), range(5)).list()
            for i in range(len(self.done_indices)):
                self.done_indices[i] -= amount
        
            
    def index_valid(self, i):
        bounded = i < self.real_capacity - 2
        safe = (i-2 < self.position) or (i > self.position + STATE_DEPTH - 1)
        return safe and bounded

    def __len__(self):
        ''' Current size or replay memory '''
        return self.position
        
        
class DQN(nn.Module):
    '''
    Define a fully-connected neural network
    '''

    def __init__(self, input_size, output_size, hidden_sizes,
                    hidden_activation = F.relu):
        '''
        Instantiates DQN
        Position Arguments:
        input_size : (int) size of state tuple
        output_size : (int) size of discrete actions
        hidden_sizes : (list of ints) sizes of hidden layer outputs
        Keyword Arguments:
        hidden_activation : (torch functional) nonlinear activations
        '''
        super(DQN, self).__init__()
        self.hidden_activation = hidden_activation
        self.in_shape = (IMG_DEPTH*STATE_DEPTH, 84, 84)        
        self.conv1 = nn.Conv2d(STATE_DEPTH,
                                FILTERS_1,
                                kernel_size=8,
                                padding=4,
                                stride=4)
                                
        self.conv2 = nn.Conv2d(in_channels=FILTERS_1, 
                                out_channels = FILTERS_2,
                                kernel_size = 4,
                                padding=2,
                                stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=FILTERS_2,
                                out_channels=FILTERS_3,
                                kernel_size=3,
                                padding=1,
                                stride=1)
        
        """
        self.conv1.weight.data.normal_(0.0, 1.0)
        self.conv1.bias.data.xavier_uniform_()
        self.conv2.weight.data.xavier_uniform_()
        self.conv2.bias.data.xavier_uniform_()
        self.conv3.weight.data.xavier_uniform_()
        self.conv3.bias.data.xavier_uniform_()
        """

        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
        nn.init.xavier_uniform(self.conv3.weight.data)

        self.n_size = self.conv_output(self.in_shape)
        
        
        sizes = [self.n_size] + hidden_sizes + [output_size]
        self.num_layers = len(hidden_sizes) + 1 # hidden layers + output_layer
        self.lin_layers = nn.ModuleList()

        for l in range(self.num_layers):
            self.lin_layers.append(nn.Linear(sizes[l], sizes[l+1]))
            nn.init.xavier_uniform(self.lin_layers[-1].weight.data)

    def conv_output(self, shape):
        inp = Variable(torch.rand(1, *shape))
        out_f = self.forward_features(inp)
        n_size = out_f.data.view(1, -1).size(1)
        return n_size

    def forward_features(self, x):
        x = self.hidden_activation(self.conv1(x))
        #x = F.max_pool3d(x, (2, 2, 1), (2, 2, 1))
        x = self.hidden_activation(self.conv2(x))
        #x = F.max_pool3d(x, (2, 2, 1), (2, 2, 1))
        x = self.hidden_activation(self.conv3(x))
        return x

    def forward(self, x):
        '''
        Given batch of states outputs Q(s, a) for states s in batch for all
        actions a.
        Positional Arguments:
        x : (Variable FloatTensor) tensor of states of size (batch_size, input_size)
        Return:
        q : (Variable FloatTensor) tensor of Q values of size (batch_size, output_size)
        '''
        
        #self.conv1.double()
        #self.conv2.double()
        
        x = self.forward_features(x)
        x = x.view(-1, self.n_size)
        h = x
        for l in range(self.num_layers - 1):
            #self.lin_layers[l].double()
            h = self.hidden_activation(self.lin_layers[l](h))
        q = self.lin_layers[-1](h)
        return q
        
        
class BreakoutAgent():
    '''
    Defines cartpole agent
    '''

    def __init__(self, num_episodes = 50000, discount = 0.99, epsilon_max = 1.0,
                epsilon_min = 0.1, epsilon_decay = 1000000, lr = 0.00025,
                batch_size = 32, copy_frequency = 10000):
        '''
        Instantiates DQN agent
        Keyword Arguments:
        num_episodes : (int) number of episodes to run agent
        discount: (float) discount factor (should be <= 1)
        epsilon_max : (float) initial epsilon. Epsilon controls how often agent selects
        random action given the state
        epsilon_min : (float) final epsilon
        epsilon_decay : (float) controls rate of epsilon decay.
        lr : (float) learning_rate
        batch_size : (int) size of batch sampled from replay memory.
        copy_frequency : (int) copy after a certain number of time steps
        '''
        
        # Added because of a memory leak bug in torch's backend
        # torch.backends.cudnn.enabled = False
        
        # Save relevant hyperparameters
        self.use_cuda = torch.cuda.is_available()
        #self.use_cuda = False
        self.lr = lr
        self.lr_min = lr / 2
        self.lr_decay_wavelength = 2500000
        
        self.num_episodes = num_episodes
        self.discount = discount
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.copy_frequency = copy_frequency
        self.loss = nn.SmoothL1Loss()
    
        # Instantiate replay memory, DQN, target DQN, optimizer, and gym environment
        self.memory = ReplayMemory()
        
        #self.env = gym.make('Breakout-v0')
        self.env_name = 'SpaceInvaders-v4'
        self.env = gym.make(self.env_name)
        self.action_space = range(self.env.action_space.n)
        self.obs_space = Breakout_obs_space()
        
        print(self.env.action_space, self.env.observation_space)
        
        self.model = DQN(self.obs_space[0] * self.obs_space[1] * self.obs_space[2], len(self.action_space), [256])
        if (self.use_cuda):
            self.model = self.model.cuda()
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, eps=1e-2, momentum=0.95, alpha=0.95)
        #self.optimizer = optim.Adadelta(self.model.parameters(), rho=0.9)
        self.train_freq = 4
        self.errors = []
        self.replay_mem_size = self.memory.capacity
        self.mem_init_size = 50
        self.action_repeat = 1
        self.no_op_max=30
        
        self.generate_replay_mem(self.mem_init_size)
 
    def select_action(self, state, steps_done = 0, explore = True):
        '''
        Given state returns an action by either randomly choosing an action or
        choosing an action that maximizes the expected reward (Q(s, a)).
        Return:
        action : (int) action choosen from state
        '''
        sample = random.random()
        #epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
        #    math.exp(-1. * steps_done / self.epsilon_decay)
        
        
        if (steps_done > self.epsilon_decay):
            epsilon = self.epsilon_min
        else:
            epsilon = self.epsilon_max - (self.epsilon_max - self.epsilon_min) * steps_done / self.epsilon_decay
        #epsilon = 0.0
        # With prob 1 - epsilon choose action to max Q
        if sample > epsilon or not explore:
            
            state = state.reshape((1, 1, 84, 84))

            aug_state = self.augment(state) / 256.0
            s = (aug_state)
            if (self.use_cuda):
                s = torch.from_numpy(s).type(torch.FloatTensor).cuda()
            else:
                s = torch.from_numpy(s).type(torch.FloatTensor)
                
            maxQ, argmax = torch.max(self.model(Variable(s, volatile = True)), dim = 1)
            s = None
            return argmax.data[0]

        # With prob epsilon choose action randomly
        else:
            a = random.randint(0, len(self.action_space)-1)
            return a
        
    def augment(self, curr_state, isnext=False, cs=None, ind=-1, location=None):
        if (STATE_DEPTH == 1):
            return curr_state
        s = curr_state.shape
        if (ind == -1):
            index = len(self.memory)-1
        else:
            index = ind-1
        
        if (type(location) != type(None)):
            output = location
        else:
            output = np.zeros((1, STATE_DEPTH, 84, 84))
        output[0,range(IMG_DEPTH),:,:] = curr_state
        i = 1
            
        counter = STATE_DEPTH-1
        previous = []
        past_start = False
        if (isnext):
            output[0, range(i*IMG_DEPTH, (i+1)*IMG_DEPTH),:,:] = cs
            i += 1
            counter -= 1
        
        
        while (counter > 0):
            if (index < 0 or index >= len(self.memory)):
                break
            prev = self.memory.memory[index]
            
            if (not prev.nonterminal):
                #curr_state = np.concatenate([curr_state, np.zeros((1, 1, s[2], s[3], IMG_DEPTH * counter))], 4)
                break
            else:
                #curr_state = np.concatenate([curr_state, prev[0]], 4);
                output[0,range(i*IMG_DEPTH, (i+1)*IMG_DEPTH),:,:] = prev.state
                
            i += 1
            index -= 1
            counter -= 1
        return output
        
    def group_augment(self, states, isnext=False, cs=None, indices=[]):
        if (STATE_DEPTH == 1):
            return np.concatenate(states)
        s = states[0].shape
        length = len(states)
        if (len(indices) == 0):
            indices = [-1 for i in range(len(states))]
        outputs = np.zeros((length, IMG_DEPTH * STATE_DEPTH, 84, 84))
        if (not isnext):
            for i in range(len(states)):
                outputs[i,:,:,:] = self.augment(states[i], ind=indices[i], location=outputs[[i],:,:,:])
        else:
            for i in range(len(states)):
                outputs[i,:,:] = self.augment(states[i], isnext=True, cs=cs[i], ind=indices[i], location=outputs[[i],:,:,:])
        return outputs / 256.0

    def train(self, show_plot = True, training=True, num_episodes=1000):
        '''
        Trains the cartpole agent.
        Keyword Arguments:
        show_plot : (boolean) indicates whether duration curve is plotted
        '''
        steps_done = 0
        durations = []
        scores = []
        if (training):
            num_episodes = self.num_episodes
        #for ep in range(num_episodes):
        curr_a = 0
        train_steps = 0
        num_saves = 0
        
        # learning rate decay
        lambda1 = lambda epoch: self.lr_min + (self.lr_decay_wavelength - train_steps) / self.lr_decay_wavelength
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lambda1])
        
        while (train_steps < 50 * self.epsilon_decay):
            
            
            curr_lives = -2
        
        
            state = self.env.reset()
            state = state.reshape((3, 210, 160))
            state = self.convert_to_grayscale(self.down_sample(state))
            done = False
            duration = 0
            curr_score = 0
            if (steps_done > 0):
                self.memory.done_indices.append(len(self.memory))
           
            #self.memory.purge()
            while not done:
            
                # handles learning rate decay
                if (train_steps < self.lr_decay_wavelength and train_steps % 1000 == 0):
                    scheduler.step()
            
                # Select action and take step
                self.env.render()
                #self.memory.states = np.concatenate([self.memory.states, state], 0)
                #aug_state = self.augment(state)
                if (steps_done % self.action_repeat == 0):
                    action = self.select_action(state, train_steps)
                    curr_a = action
                else:
                    action = curr_a
                next_state, reward, done, info = self.env.step(action)
                lives = info['ale.lives']
                reward = self.regularize_reward(reward)
                r = reward

                # Convert s, a, r, s', d to tensors
                next_state = next_state.reshape((3, 210, 160))
                next_state = self.convert_to_grayscale(self.down_sample(next_state))
                action = action
                reward = reward
                nonterminal = not done
                if (curr_lives - lives == 1):
                    nonterminal = False
                curr_lives = lives

                # Remember s, a, r, s', d
                #self.memory.push((state, action, next_state, reward, nonterminal))
                episode = ((state, action, None, reward, nonterminal))
                self.memory.push((state, action, None, reward, nonterminal))
                steps_done += 1
                state = next_state
                duration += 1
                curr_score += r

                # Sample from replay memory if full memory is full capacity
                
                if steps_done % self.train_freq == 0 and training:
                    train_steps += 1
                    #batch = self.memory.sample(self.batch_size)
                    #batch = Transition(*zip(*batch))
                    batch, indices = self.memory.sample(self.batch_size)
                    batch = Transition(*zip(*batch))
                    x = (self.group_augment(batch.state, indices=indices)) 
                    #y = self.group_augment(batch.next_state, isnext=True, cs=batch.state, indices=indices) / 256.0
                    n_states = self.next_states(indices)
                    y = (self.group_augment(n_states, isnext=True, cs=batch.state, indices=indices)) 
                    outs = 0
                    if (self.use_cuda):
                        state_batch = Variable(torch.from_numpy(x).type(torch.FloatTensor)).cuda()
                        n = state_batch.data.shape[0]
                        actions = np.array(batch.action).reshape((n, 1))
                        action_batch = Variable(torch.from_numpy(actions)).cuda()
                        next_state_batch = Variable(torch.from_numpy(y).type(torch.FloatTensor), volatile = True).cuda()
                        rewards = np.array(batch.reward).reshape((n, 1))
                        reward_batch = Variable(torch.from_numpy(rewards).type(torch.FloatTensor)).cuda()
                        
                        nonterminal = np.array(batch.nonterminal, dtype=np.uint8).reshape((n, 1))
                        nonterminal_mask = Variable(torch.from_numpy(nonterminal).type(torch.ByteTensor)).cuda()

                        # Predict Q(s, a) for s in batch
                        outs = self.model(state_batch)
                        q_batch = outs.gather(1, action_batch)

                        # Calcuate target values
                        # if terminal state, then target = rewards
                        # else target = r(s, a) + discount * max_a Q(s', a) where s' is
                        # next state
                        next_state_values = Variable(torch.zeros(n,1), volatile = True).cuda()
                    else:
                        state_batch = Variable(torch.from_numpy(x).type(torch.FloatTensor))
                        n = state_batch.data.shape[0]
                        actions = np.array(batch.action).reshape((n, 1))
                        action_batch = Variable(torch.from_numpy(actions))
                        next_state_batch = Variable(torch.from_numpy(y).type(torch.FloatTensor), volatile = True)
                        rewards = np.array(batch.reward).reshape((n, 1))
                        reward_batch = Variable(torch.from_numpy(rewards).type(torch.FloatTensor))
                        
                        nonterminal = np.array(batch.nonterminal, dtype=np.uint8).reshape((n, 1))
                        nonterminal_mask = Variable(torch.from_numpy(nonterminal).type(torch.ByteTensor))

                        # Predict Q(s, a) for s in batch
                        outs = self.model(state_batch)
                        q_batch = outs.gather(1, action_batch)

                        # Calcuate target values
                        # if terminal state, then target = rewards
                        # else target = r(s, a) + discount * max_a Q(s', a) where s' is
                        # next state
                        next_state_values = Variable(torch.zeros(n,1), volatile = True)
                        
                    action_indices = torch.nonzero(nonterminal_mask)[:,0]
                    preds = self.target_model(next_state_batch[action_indices])
                    next_state_values[nonterminal_mask], _ = torch.max(
                                preds,
                                dim = 1)
                    # Make sure the final loss is not volatile
                    next_state_values.volatile = False
                    next_state_values = next_state_values * self.discount +  reward_batch
                    # Define loss function and optimize
                    loss = self.loss(q_batch, next_state_values.detach())
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    #torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    
                    """
                    if (self.use_cuda):
                        torch.cuda.empty_cache()
                    """
                    if (self.use_cuda):
                        l = loss.data[0]
                    else:
                        l = loss.data[0]
                    self.errors.append(l)
                    #q_sample = q_batch.data[0][0]
                    sample_qs = outs.data[0].cpu().numpy()
                    target = next_state_values.data[0]
                    self.print_statistics(len(self.errors), l, sample_qs, target)
                    
                    """
                    del batch
                    del state_batch
                    del action_batch
                    del next_state_batch
                    del reward_batch
                    del nonterminal_mask
                    del next_state_values
                    del action_indices
                    del preds
                    del q_batch
                    del loss
                    """
                    
                    if (len(self.errors) % 200000 == 0):
                        #self.model.module.save_state_dict('mytraining.pt')
                        #torch.save(self.model.module.state_dict(), 'mytraining.pt')
                        folder_name = self.env_name + '_models'
                        if not os.path.exists(folder_name):
                            os.mkdir(folder_name)
                        filename = folder_name + '/mytraining.pt' + str(num_saves)
                        num_saves += 1
                        torch.save(self.model.state_dict(), filename)
                
                # Copy to target network
                # Most likely unneeded for cart pole, but targets networks are used
                # generally in DQN.
                
                if steps_done % self.copy_frequency == 0 and len(self.memory) >= self.mem_init_size:
                    del self.target_model
                    self.target_model = copy.deepcopy(self.model)
                    
                
                # Plot durations
                if done and show_plot and len(self.errors) > 0:
                    print("Game %d lasted %d frames with score %d" % (len(scores), duration, curr_score))
                    durations.append(duration)
                    scores.append(curr_score)
                    plot_period = 50
                    if (len(scores) % plot_period) == 0:
                        self.plot_scores(scores, plot_period)
                    duration = 0
                    curr_score = 0
                    self.env.reset()
                
              


    def generate_replay_mem(self, mem_len):
        num_steps = 0
        num_games = 1
        while(num_steps < mem_len):
            
            no_op_num = 0
            
            state = self.env.reset()
            state = state.reshape((3, 210, 160))
            state = self.convert_to_grayscale(self.down_sample(state))
            done = False
            print("Beginning game %d" % num_games)
            curr_lives = -2
            while not done:
                if (no_op_num > self.no_op_max):
                    action = random.randint(1, len(self.action_space)-1)
                action = random.randint(0, len(self.action_space)-1)
                if (action == 0):
                    no_op_num += 1
                
                #self.env.render()
                next_state, reward, done, info = self.env.step(action)
                lives = info['ale.lives']
                reward = self.regularize_reward(reward)
                # Convert s, a, r, s', d to tensors
                next_state = next_state.reshape((3, 210, 160))
                next_state = self.convert_to_grayscale(self.down_sample(next_state))
                action = action
                reward = reward
                nonterminal = not done
                if (curr_lives - lives == 1):
                    nonterminal = False
                curr_lives = lives

                # Remember s, a, r, s', d
                #self.memory.push((state, action, next_state, reward, nonterminal))
                episode = (state, action, None, reward, nonterminal)
                self.memory.push(episode)
                state = next_state
                num_steps += 1
                
            self.memory.done_indices.append(num_steps)
            num_games += 1
                    
    def next_states(self, indices):
        return self.memory.next_states(indices)
                    
    def print_statistics(self, iter_num, loss, sample_qs, sample_target):
        print("Loss at iteration %d is %f. Vals: %s. Targets: %f" % (iter_num, loss, np.array_str(sample_qs), sample_target)),
	

    def displayStack(self, state):
        state = state.reshape(INPUT_SHAPE)
        self.displayImage(state[0,:,:])
        self.displayImage(state[1,:,:])
        self.displayImage(state[2,:,:])
        self.displayImage(state[3,:,:])
        

    def displayImage(self, image):
        plt.imshow(image)
        plt.show()

    def plot_durations(self, durations):
        '''
        Plots duration curve
        Positional Arguments:
        durations : (list of ints) duration for every episode
        '''
        plt.figure(1)
        plt.clf()
        durations_a = np.array(durations)
        plt.title('Training...')
        plt.xlabel('Game Number')
        plt.ylabel('Duration')
        plt.plot(durations_a)
        plt.pause(0.001)  # pause a bit so that plots are updated
        
    def plot_scores(self, scores, plot_period):
        plt.figure(1)
        plt.clf()
        scores_a = np.array(scores)
        scores_average = np.mean(scores_a.reshape(-1, plot_period), axis=1)
        plt.title('Training...')
        plt.xlabel('Game Epoch')
        plt.ylabel('Score')
        plt.plot(scores_average)
        plt.show(block=False)
        plt.pause(0.01)  # pause a bit so that plots are updated

    def run_and_visualize(self):
        ''' Runs and visualizes the cartpole agents. '''
                
        folder_name = self.env_name + '_models'
        full_name = folder_name + '/mytraining.pt'
        self.model.load_state_dict(torch.load(full_name+'50'))
        self.model = self.model.cuda()
        
        self.env.reset()
        
        num_models = 222
        num_games = 10
        for j in range(num_models):
            self.model.load_state_dict(torch.load(full_name + str(j)))
            self.model = self.model.cuda()
            score_total = 0
            t_total = 0
            for i in range(num_games):
                #self.env.reset()
                state = self.env.reset()
                actions = []
                done = False
                t = 0
                score = 0
                while not done:
                    state = state.reshape((3, 210, 160))
                    state = self.convert_to_grayscale(self.down_sample(state))
                    action = self.select_action(state, explore = False)
                    if (random.random() > 0.98):
                        action = random.randint(0, len(self.action_space)-1)
                    #time.sleep(0.01)
                    next_state, reward, done, _ = self.env.step(action)
                    t += 1
                    score += reward
                    nonterminal = not done
                    episode = (state, action, None, reward, nonterminal)
                    self.memory.push(episode)
                    self.env.render()
                    state = next_state
                    if done:
                        print("Model {} finished after {} timesteps with score {}".format(j, t+1, score))
                        score_total += score
                        t_total += t
                        break
                        
            print("Model {} had an average score of {} with average length {}".format(j, score_total/num_games, t_total/num_games))
                        
    def generate_video(self):
        
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('breakout.avi', fourcc, 50, (160, 210), isColor=True)
        
        num_models = 28
        num_games = 5
        
        games = []
        scores = []
        
        for i in range(8, 20):
            self.model.load_state_dict(torch.load('models/mytraining.pt' + str(i)))
            self.model = self.model.cuda()
            
            for j in range(num_games):
                curr_game = []
                
                state = self.env.reset()
                score = 0
                t = 0
                done = False
                while not done:
                    orig_state = np.flip(state, 2)
                    curr_game.append(orig_state)
                    state = state.reshape((3, 210, 160))
                   
                    state = self.convert_to_grayscale(self.down_sample(state))
                    action = self.select_action(state, explore = False)
                    if (random.random() > 0.97):
                        action = random.randint(0, len(self.action_space)-1)
                    next_state, reward, done, _ = self.env.step(action)
                    score += reward
                    t += 1
                    #self.env.render()
                    nonterminal = not done
                    episode = (state, action, None, reward, nonterminal)
                    self.memory.push(episode)
                    state = next_state                    
                    if done:
                        print("Model {} finished after {} timesteps with score {}".format(i, t+1, score))
                        break
                if ((score >= 420 and t <= 3000) or score >= 450):
                    games.append(curr_game)
                    scores.append(score)
                    if (len(games) >= 1):
                        break
            if (len(games) >= 1):
                break
        
        score_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        for i in range(1):
            game = games[score_indices[i]]
            for j in range(len(game)):
                out.write(game[j])
                            
        out.release()
        
                    
            
                    
    def convert_to_grayscale(self, state):
        state = np.mean(state, axis=0).astype(np.uint8).reshape((1, 84, 84))
        return state
        
    def down_sample(self, img):
        top=34
        bot=34+160
        left=0
        right=160
        if (img.shape != (210, 160, 3)):
            img = img.reshape((210, 160, 3))
        img = img[top:bot, left:right, :]
        img = scipy.misc.imresize(img, (84, 84, 3), interp='nearest')
        img = np.swapaxes(np.swapaxes(img, 0, 1), 0, 2)
        return img
    
    def regularize_reward(self, r):
        """
        if (r > 0):
            return 1
        return 0
        """
        return r
                    
def Breakout_action_space():
    return range(4)
    
def Breakout_obs_space():
    return (210, 160, 3)

def main():
    cpa = BreakoutAgent()
    print(cpa.model)
    #cpa.train()
    cpa.run_and_visualize()
    #cpa.generate_video()
    # cpa.model.load_state_dict(torch.load('mytraining.pt'))
    #cpa.train(training=False, num_episodes=100000)

if __name__ == '__main__':
    main()


