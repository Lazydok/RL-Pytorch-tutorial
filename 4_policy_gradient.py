import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import gym
from torch.autograd import Variable
import random

HIDDEN_LAYER = 24  # NN hidden layer size
LR = 0.01
GAMMA = 0.99

INPUT_SIZE = 4
OUTPUT_SIZE = 2

ENV = gym.make('CartPole-v0').unwrapped
HISTORY = []

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER)
        # nn.init.xavier_uniform(self.l1.weight)
        self.l2 = nn.Linear(HIDDEN_LAYER, OUTPUT_SIZE)
        # nn.init.xavier_uniform(self.l2.weight)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x

model = Network()

use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

optim = torch.optim.Adam(model.parameters(), lr=LR)

def discount_rewards(r):
    discounted_r = torch.zeros(r.size())
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add

    return discounted_r

def run_episode(net, e, env):
    state = env.reset()
    reward_sum = 0
    xs = FloatTensor([])
    ys = FloatTensor([])
    rewards = FloatTensor([])
    steps = 0

    while True:
        # env.render()

        x = FloatTensor([state])
        xs = torch.cat([xs, x])

        action_prob = net(Variable(x))

        # select an action depends on probability
        action = 0 if random.random() < action_prob.data[0][0] else 1

        y = FloatTensor([[1, 0]] if action == 0 else [[0, 1]])
        ys = torch.cat([ys, y])

        state, reward, done, _ = env.step(action)
        rewards = torch.cat([rewards, FloatTensor([[reward]])])
        reward_sum += reward
        steps += 1

        if done or steps >= 500:
            adv = discount_rewards(rewards)
            # adv = (adv - adv.mean())
            adv = (adv - adv.mean())/(adv.std() + 1e-7)
            # print(adv)
            loss = learn(xs, ys, adv)
            HISTORY.append(reward_sum)
            print("[Episode {:>5}]  steps: {:>5} loss: {:>5}".format(e, steps, loss))
            if sum(HISTORY[-5:])/5 > 490:
                return True
            else:
                return False

def learn(x, y, adv):
    # Loss function, ∑ Ai*logp(yi∣xi), but we need fake lable Y due to autodiff
    action_pred = model(Variable(x))
    y = Variable(y, requires_grad=True)
    adv = Variable(adv).cuda()
    # print(action_pred)
    log_lik = -y * torch.log(action_pred)
    # print(y)
    log_lik_adv = log_lik * adv
    # print(torch.sum(log_lik_adv, 1))
    loss = torch.sum(log_lik_adv, 1).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.data[0]


for e in range(10000):
    complete = run_episode(model, e, ENV)

    if complete:
        print('complete...!')
        break

import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

def botPlay(env):
    state = env.reset()
    steps = 0
    frames = []
    while True:
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        action = torch.max(model(Variable(FloatTensor([state]))), 1)[1].data[0]
        next_state, reward, done, _ = env.step(action)

        state = next_state
        steps += 1

        if done or steps >= 1000:
            break

    clip = ImageSequenceClip(frames, fps=20)
    clip.write_gif('4_policy_gradient_play.gif', fps=20)

def plot_durations(d):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(d)

    plt.savefig('4_policy_gradient_score.png')

botPlay(ENV)
plot_durations(HISTORY)