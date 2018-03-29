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
from operator import itemgetter

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'nonterminal'))

STATE_DEPTH = 4
IMG_DEPTH = 1

######################################################################
# Replay Memory
# Stores state, action, next_state, reward, done

class ReplayMemory():
    def __init__(self, capacity = 600000):
        ''' Initializes empty replay memory '''
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.done_indices = []
        self.sensitive_indices = []

    def push(self, item):
        ''' Stores item in replay memory '''
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        ''' Samples item from replay memory '''
        #if (random.random() > 0.8 and len(self.sensitive_indices) > batch_size):
        #    indices = random.sample(self.sensitive_indices, batch_size)
        #else:
        indices = random.sample(range(len(self.memory)), batch_size)
        samples = np.array([self.memory[i] for i in indices if i < len(self.memory)])
       
        
        return samples, indices
    
    def purge(self):
        if (len(self.memory) > 20000):
            amount = self.done_indices[5]
            self.memory = np.delete(np.array(self.memory), range(amount)).list()
            self.states = np.delete(self.states, range(amount))
            self.done_indices = np.delete(np.array(self.done_indices), range(5)).list()
            for i in range(len(self.done_indices)):
                self.done_indices[i] -= amount
        
    def augment_sample(self, sample):
        for i in range(len(sample)):
            s1 = self.memory[i]

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
        self.in_shape = (1, 105, 80, IMG_DEPTH*STATE_DEPTH)        
        self.conv1 = nn.Conv3d(in_channels=1,
                                out_channels=16,
                                kernel_size=(8, 8, IMG_DEPTH*STATE_DEPTH),
                                padding=(4, 4, 0),
                                stride=4)
                                
        self.conv2 = nn.Conv3d(in_channels=16, 
                                out_channels = 32,
                                kernel_size = (4, 4, 1),
                                padding=(2, 2, 0),
                                stride=2)
        
        self.n_size = self.conv_output(self.in_shape)
        
        
        sizes = [self.n_size] + hidden_sizes + [output_size]
        self.num_layers = len(hidden_sizes) + 1 # hidden layers + output_layer
        self.lin_layers = nn.ModuleList()

        for l in range(self.num_layers):
            self.lin_layers.append(nn.Linear(sizes[l], sizes[l+1]))
            self.lin_layers[-1].weight.data.uniform_(-0.01, 0.01)
            self.lin_layers[-1].bias.data.uniform_(-0.01, 0.01)

    def conv_output(self, shape):
        inp = Variable(torch.rand(1, *shape))
        out_f = self.forward_features(inp)
        n_size = out_f.data.view(1, -1).size(1)
        return n_size

    def forward_features(self, x):
        x = self.hidden_activation(self.conv1(x))
        x = F.max_pool3d(x, (2, 2, 1), (2, 2, 1))
        x = self.hidden_activation(self.conv2(x))
        x = F.max_pool3d(x, (2, 2, 1), (2, 2, 1))
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
                epsilon_min = 0.05, epsilon_decay = 10e6, lr = 2.5e-4,
                batch_size = 32, copy_frequency = 500):
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
        self.use_cuda = torch.cuda.is_available()
        #self.use_cuda = False
        
        self.num_episodes = num_episodes
        self.discount = discount
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.copy_frequency = copy_frequency
    
        # Instantiate replay memory, DQN, target DQN, optimizer, and gym environment
        self.memory = ReplayMemory()
        
        #self.env = gym.make('Breakout-v0')
        self.env = gym.make('BreakoutDeterministic-v4')
        self.action_space = Breakout_action_space()
        self.obs_space = Breakout_obs_space()
        
        print(self.env.action_space, self.env.observation_space)
        
        self.model = DQN(self.obs_space[0] * self.obs_space[1] * self.obs_space[2], len(self.action_space), [256])
        if (self.use_cuda):
            self.model = torch.nn.DataParallel(self.model).cuda()
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optim.SGD(self.model.parameters(), lr = lr, momentum=0.95)
        self.train_freq = 4
        self.errors = []
        self.replay_mem_size = self.memory.capacity
        self.mem_init_size = 50000
 
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
            epsilon = 0.1
        else:
            epsilon = self.epsilon_max - (self.epsilon_max - self.epsilon_min) * steps_done / self.epsilon_decay
        #epsilon = 0.0
        # With prob 1 - epsilon choose action to max Q
        if sample > epsilon or not explore:
            state = state / 256.0
            if (self.use_cuda):
                state = torch.from_numpy(state).type(torch.FloatTensor).cuda()
            else:
                state = torch.from_numpy(state).type(torch.FloatTensor)
                
            maxQ, argmax = torch.max(self.model(Variable(state, volatile = True)), dim = 1)
            return argmax.data[0]

        # With prob epsilon choose action randomly
        else:
            return random.randint(0, len(self.action_space)-1)
        
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
            output = np.zeros((1, 1, s[2], s[3], IMG_DEPTH*STATE_DEPTH))
        output[:,:,:,:,range(IMG_DEPTH)] = curr_state
        i = 1
            
        counter = STATE_DEPTH-1
        previous = []
        past_start = False
        if (isnext):
            #curr_state = np.concatenate([curr_state, cs], 4)
            output[:,:,:,:,range(i*IMG_DEPTH, (i+1)*IMG_DEPTH)] = cs
            i += 1
            counter -= 1
        while (counter > 0):
            if (index < 0 or index >= len(self.memory)):
                #curr_state = np.concatenate([curr_state, np.zeros((1, 1, s[2], s[3], IMG_DEPTH * counter))], 4)
                break
            prev = self.memory.memory[index]
            
            if (not prev[4][0]):
                #curr_state = np.concatenate([curr_state, np.zeros((1, 1, s[2], s[3], IMG_DEPTH * counter))], 4)
                break
            else:
                #curr_state = np.concatenate([curr_state, prev[0]], 4);
                output[:,:,:,:,range(i*IMG_DEPTH, (i+1)*IMG_DEPTH)] = prev[0]
                
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
        outputs = np.zeros((length, 1, s[2], s[3], IMG_DEPTH * STATE_DEPTH))
        if (not isnext):
            for i in range(len(states)):
                outputs[i,:,:,:,:] = self.augment(states[i], ind=indices[i], location=outputs[[i],:,:,:,:])
        else:
            for i in range(len(states)):
                outputs[i,:,:,:,:] = self.augment(states[i], isnext=True, cs=cs[i], ind=indices[i], location=outputs[[i],:,:,:,:])
        return outputs

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
        while (steps_done < 10 * self.epsilon_decay):
            state = self.env.reset()
            state = state.reshape((1, 1, 210, 160, 3))
            state = self.down_sample(self.convert_to_grayscale(state))
            done = False
            duration = 0
            curr_score = 0
            self.memory.done_indices.append(steps_done)
            print("Beginning game %d" % len(self.memory.done_indices))
            #self.memory.purge()
            while not done:
                # Select action and take step
                self.env.render()
                #self.memory.states = np.concatenate([self.memory.states, state], 0)
                aug_state = self.augment(state)
                action = self.select_action(aug_state, steps_done)
                next_state, reward, done, _ = self.env.step(action)

                if (done):
                    reward -= 1
                    
                r = reward

                # Convert s, a, r, s', d to tensors
                next_state = next_state.reshape((1, 1, 210, 160, 3))
                next_state = self.down_sample(self.convert_to_grayscale(next_state))
                action = torch.LongTensor([[action]])
                reward = torch.FloatTensor([reward])
                nonterminal = torch.ByteTensor([not done])

                # Remember s, a, r, s', d
                self.memory.push((state, action, next_state, reward, nonterminal))
                if (len(self.memory) > self.mem_init_size):
                    steps_done += 1
                state = next_state
                duration += 1
                curr_score += r


                # Sample from replay memory if full memory is full capacity
                if len(self.memory) >= self.mem_init_size and steps_done % self.train_freq == 0 and training:
                    #batch = self.memory.sample(self.batch_size)
                    #batch = Transition(*zip(*batch))
                    batch, indices = self.memory.sample(self.batch_size)
                    batch = Transition(*zip(*batch))
                    x = self.group_augment(batch.state, indices=indices) / 256.0
                    y = self.group_augment(batch.next_state, isnext=True, cs=batch.state, indices=indices) / 256.0
                    
                    #self.displayStack(x[0,:,:,:,:])
                    if (self.use_cuda):
                        state_batch = Variable(torch.from_numpy(x).type(torch.FloatTensor)).cuda()
                        action_batch = Variable(torch.cat(batch.action)).cuda()
                        next_state_batch = Variable(torch.from_numpy(y).type(torch.FloatTensor), volatile = True).cuda()
                        reward_batch = Variable(torch.cat(batch.reward)).cuda()
                        nonterminal_mask = torch.cat(batch.nonterminal).cuda()

                        # Predict Q(s, a) for s in batch
                        q_batch = self.model(state_batch).gather(1, action_batch)

                        # Calcuate target values
                        # if terminal state, then target = rewards
                        # else target = r(s, a) + discount * max_a Q(s', a) where s' is
                        # next state
                        next_state_values = Variable(torch.zeros(self.batch_size)).cuda()
                    else:
                        state_batch = Variable(torch.from_numpy(x).type(torch.FloatTensor))
                        action_batch = Variable(torch.cat(batch.action))
                        next_state_batch = Variable(torch.from_numpy(y).type(torch.FloatTensor), volatile = True)
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
                    preds = self.target_model(next_state_batch[indices])
                    # print(indices.shape, preds.shape)
                    #print(next_state_values[nonterminal_mask].data.shape, torch.max(preds, dim = 1)[0].data.shape)
                    next_state_values[nonterminal_mask], _ = torch.max(
                                preds,
                                dim = 1)

                    # Make sure the final loss is not volatile
                    next_state_values.volatile = False
                    next_state_values = next_state_values * self.discount +  reward_batch

                    # Define loss function and optimize
                    loss = F.mse_loss(q_batch, next_state_values)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    if (self.use_cuda):
                        l = loss[0]
                    else:
                        l = loss.data[0]
                    self.errors.append(l)
                    self.print_statistics(len(self.errors), l)

                # Copy to target network
                # Most likely unneeded for cart pole, but targets networks are used
                # generally in DQN.
                if len(self.errors) % self.copy_frequency == 0:
                    self.target_model = copy.deepcopy(self.model)

                # Plot durations
                if done and show_plot and len(self.errors) > 0 and len(durations) % 10 == 5:
                    durations.append(duration)
                    scores.append(curr_score)
                    self.plot_scores(scores)
                    duration = 0
                    curr_score = 0
                    self.env.reset()
                
                if (r != 0):
                    self.memory.sensitive_indices.append(steps_done)
            
                    
    def print_statistics(self, iter_num, loss):
        print("Loss at iteration %d is %f" % (iter_num, loss))

    def displayStack(self, state):
        state = state.reshape((210, 160, STATE_DEPTH))
        self.displayImage(state[:,:,0])
        self.displayImage(state[:,:,1])
        self.displayImage(state[:,:,2])
        self.displayImage(state[:,:,3])
        

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
        
    def plot_scores(self, scores):
        plt.figure(1)
        plt.clf()
        scores_a = np.array(scores)
        plt.title('Training...')
        plt.xlabel('Game Number')
        plt.ylabel('Score')
        plt.plot(scores_a)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def run_and_visualize(self):
        ''' Runs and visualizes the cartpole agents. '''
        self.env.reset()
        for i in range(50):
            #self.env.reset()
            print ("Running and visualizing now . . .")
            state = self.env.reset()
            state = torch.from_numpy(state.reshape((1, 1, 210, 160, 3))).type(torch.FloatTensor)
            actions = []
            done = False
            t = 0
            while not done:
                print ("loop is iterating", t)
                action = self.select_action(state, explore = False)
                state, reward, done, _ = self.env.step(action)
                state = torch.from_numpy(state.reshape((1, 1, 210, 160, IMG_DEPTH))).type(torch.FloatTensor)
                t += 1
                time.sleep(0.05)
                self.env.render()
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
                    
    def convert_to_grayscale(self, state):
        gray_state = np.mean(state, axis=4).reshape((1, 1, 210, 160, 1)).astype(np.uint8)
        return gray_state
        
    def down_sample(self, img):
        return img[:, :, ::2, ::2, :]
                    
def Breakout_action_space():
    return range(4)
    
def Breakout_obs_space():
    return (210, 160, 3)

def main():
    cpa = BreakoutAgent()
    print(cpa.model)
    cpa.train()
    cpa.model.save_state_dict('mytraining.pt')
    # cpa.model.load_state_dict(torch.load('mytraining.pt'))
    cpa.train(training=False, num_episodes=100000)

if __name__ == '__main__':
    main()



