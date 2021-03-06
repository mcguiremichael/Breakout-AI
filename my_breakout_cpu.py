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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'nonterminal'))

######################################################################
# Replay Memory
# Stores state, action, next_state, reward, done

class ReplayMemory():
    def __init__(self, capacity = 10000):
        ''' Initializes empty replay memory '''
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, item):
        ''' Stores item in replay memory '''
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        ''' Samples item from replay memory '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        ''' Current size or replay memory '''
        return len(self.memory)
        
        
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
        self.in_shape = (1, 210, 160, 3)        
        self.conv1 = nn.Conv3d(in_channels=1,
                                out_channels=1,
                                kernel_size=(20, 20, 3),
                                padding=(5, 5, 0),
                                stride=10)
        
        self.n_size = self.conv_output(self.in_shape)
        
        
        sizes = [self.n_size] + [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(hidden_sizes) + 1 # hidden layers + output_layer
        self.lin_layers = nn.ModuleList()

        for l in range(self.num_layers+1):
            self.lin_layers.append(nn.Linear(sizes[l], sizes[l+1]))

    def conv_output(self, shape):
        inp = Variable(torch.rand(1, *shape))
        out_f = self.forward_features(inp)
        n_size = out_f.data.view(1, -1).size(1)
        return n_size

    def forward_features(self, x):
        x = self.hidden_activation(self.conv1(x))
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
        x = self.forward_features(x)
        x = x.view(-1, self.n_size)
        h = x
        for l in range(self.num_layers - 1):
            h = self.hidden_activation(self.lin_layers[l](h))
        q = self.lin_layers[-1](h)
        return q
        
        
class BreakoutAgent():
    '''
    Defines cartpole agent
    '''

    def __init__(self, num_episodes = 100, discount = 0.999, epsilon_max = 1.0,
                epsilon_min = 0.05, epsilon_decay = 200, lr = 3e-3,
                batch_size = 1, copy_frequency = 10):
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
        # Save relevant hyperparameters
        self.num_episodes = num_episodes
        self.discount = discount
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.copy_frequency = copy_frequency

        # Instantiate replay memory, DQN, target DQN, optimizer, and gym environment
        self.memory = ReplayMemory()
        
        self.env = gym.make('Breakout-v0')
        self.action_space = Breakout_action_space()
        self.obs_space = Breakout_obs_space()
        

        
        self.model = DQN(self.obs_space[0] * self.obs_space[1] * self.obs_space[2], len(self.action_space), [128, 128])
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)

    def select_action(self, state, steps_done = 0, explore = True):
        '''
        Given state returns an action by either randomly choosing an action or
        choosing an action that maximizes the expected reward (Q(s, a)).
        Return:
        action : (int) action choosen from state
        '''
        sample = random.random()
        epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
            math.exp(-1. * steps_done / self.epsilon_decay)

        # With prob 1 - epsilon choose action to max Q
        if sample > epsilon or not explore:
            maxQ, argmax = torch.max(self.model(Variable(state, volatile = True)), dim = 1)
            return argmax.data[0]

        # With prob epsilon choose action randomly
        else:
            return random.randint(0, len(self.action_space)-1)

    def train(self, show_plot = True):
        '''
        Trains the cartpole agent.
        Keyword Arguments:
        show_plot : (boolean) indicates whether duration curve is plotted
        '''
        steps_done = 0
        durations = []
        for ep in range(self.num_episodes):
            state = self.env.reset()
            state = torch.from_numpy(state.reshape((1, 1, 210, 160, 3))).type(torch.FloatTensor)
            done = False
            duration = 0
            while not done:
                print (steps_done)
                # Select action and take step
                action = self.select_action(state, steps_done)
                next_state, reward, done, _ = self.env.step(action)

                # Convert s, a, r, s', d to tensors
                next_state = torch.from_numpy(next_state.reshape((1, 1, 210, 160, 3))).type(torch.FloatTensor)
                action = torch.LongTensor([[action]])
                reward = torch.FloatTensor([reward])
                nonterminal = torch.ByteTensor([not done])

                # Remember s, a, r, s', d
                self.memory.push((state, action, next_state, reward, nonterminal))
                steps_done += 1
                state = next_state
                duration += 1

                # Sample from replay memory if full memory is full capacity
                if len(self.memory) >= self.batch_size:
                    batch = self.memory.sample(self.batch_size)
                    batch = Transition(*zip(*batch))
                    state_batch = Variable(torch.cat(batch.state))
                    action_batch = Variable(torch.cat(batch.action))
                    next_state_batch = Variable(torch.cat(batch.next_state), volatile = True)
                    reward_batch = Variable(torch.cat(batch.reward))
                    nonterminal_mask = torch.cat(batch.nonterminal)

                    # Predict Q(s, a) for s in batch
                    q_batch = self.model(state_batch).gather(1, action_batch)

                    # Calcuate target values
                    # if terminal state, then target = rewards
                    # else target = r(s, a) + discount * max_a Q(s', a) where s' is
                    # next state
                    next_state_values = Variable(torch.zeros(self.batch_size))
                    indices = torch.nonzero(nonterminal_mask).squeeze(1)
                    next_state_values[nonterminal_mask], _ = torch.max(
                                self.target_model(next_state_batch[indices]), dim = 1)

                    # Make sure the final loss is not volatile
                    next_state_values.volatile = False
                    next_state_values = next_state_values * self.discount + reward_batch

                    # Define loss function and optimize
                    loss = F.mse_loss(q_batch, next_state_values)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Copy to target network
                # Most likely unneeded for cart pole, but targets networks are used
                # generally in DQN.
                if steps_done % self.copy_frequency == 0:
                    self.target_model = copy.deepcopy(self.model)

                # Plot durations
                if done and show_plot:
                    durations.append(duration)
                    self.plot_durations(durations)
                    duration = 0

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
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_a)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def run_and_visualize(self):
        ''' Runs and visualizes the cartpole agents. '''
        state = self.env.reset()
        state = torch.FloatTensor([state])
        actions = []
        for i in range(500):
            self.env.reset()
            for t in range(1000):
                self.env.render()
                action = self.select_action(state, explore = False)
                state, reward, done, _ = self.env.step(action)
                state = torch.FloatTensor([state])
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
                    
def Breakout_action_space():
    return range(4)
    
def Breakout_obs_space():
    return (210, 160, 3)

def main():
    cpa = BreakoutAgent()
    print(cpa.model)
    cpa.train()
    cpa.run_and_visualize()

if __name__ == '__main__':
    main()

