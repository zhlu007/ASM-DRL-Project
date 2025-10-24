import math
import random
import time

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib.pyplot as plt
from matplotlib import animation

import argparse
# from utils import create_directory, plot_learning_curve

# ------------------设置参数信息----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=400)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/D3QN/')
parser.add_argument('--reward_path', type=str, default='./output_images/reward.png')
parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png')

args = parser.parse_args(args=[])

edge_count = 0
geo_count = 0
ground_count = 0


N = 3
max_hop = 7
distance_between_leo = 4481  # KM 相邻两颗卫星之间的距离
high_leo = 780  # KM leo卫星的高度
c = 3e5  # KM/s光速
tau = 0.2  # s 时间片大小
p_user = 1  # W  用户发送功率(33dbm)
p_leo = 5
r_leo = 30  # Mbps 卫星发送速率
r_geo = 90
B = 10  # MHz 信道带宽
fc = 30  # GHz 载波频率     带宽和频率是否一致
N0 = 1e-9  # W 信道噪声
# 计算信道增益h
Ad = 4.11  # 天线增益
de = 2.8  # 路径衰减
h = 10  # 信道增益
epsilon = 5e-26  # J/Hz3/s  能量系数

# variable
w = 200  # cycles/bit  处理强度
f_user = 0.2  # GHz 用户计算能力
f_leo = 2.1  # GHz leo卫星计算能力
min_task_size = 350
max_task_size = 600
flow_multiplier = 10
flow_noise = 0
lam = 0.5
cost_multiplier = 0.1
max_queue_size = 100


class Task:
    def __init__(self, index, task_size):
        self.index = index
        self.task_size = task_size  # 任务大Mb
        self.x = 0  # 卸载策略  0：本地计算  其他的在相应的卫星进行计算
        self.energy_need = 0.5
        self.delay_need = 1 - self.energy_need
        self.delay = {'send_delay': 0, 'propagate_delay': 0, 'forward_delay': 0, 'queue_delay': 0,
                      'calculate_delay': 0}  # 记录时延
        self.energy = {'send_energy': 0, 'calculate_energy': 0}  # 记录能耗
        self.cost = 0  # 代价


class ENV:
    def __init__(self, I, N):
        self.I = I  # number of users
        self.N = N  # number of leo
        self.TASK = []  # init task collection
        self.leo_queue_size = np.zeros(N)  # 初始化leo队列
        self.leo_forward_leo_queue = np.zeros(1)  # 初始化leo间队列
        self.leo_forward_geo_queue = np.zeros(1)  # 初始化leo间队列
        self.state = np.zeros(I + N + 3)  # 初始化环境状态
        self.time_slot = 1  # 时间片计数器
        self.cloud_hop = np.random.randint(0, max_hop, 1)
        self.cloud_hop[0] = 0
        self.edge_count = edge_count
        self.geo_count = geo_count
        self.ground_count = ground_count
        self.total_count = 0

    def reset(self):
        self.__init__(self.I, self.N)
        self.TASK, task_size = self.task_generate()  # 随机生成任务
        # self.cloud_hop = random.randint(0, max_hop)
        self.cloud_hop = np.random.randint(0, max_hop, 1)
        self.cloud_hop[0] = 0
        self.state = np.concatenate([task_size, self.leo_queue_size,
                                     self.leo_forward_geo_queue, self.leo_forward_leo_queue, self.cloud_hop])
        self.time_slot = 1
        return self.state

    def send_rate(self):
        # print(h)
        return B * math.log2(1 + (p_user * h) / (N0 * self.I * p_user * h))

    def task_generate(self):
        self.time_slot += 1
        TASK = []
        task_size = np.random.randint(min_task_size, max_task_size, self.I) / 1e3  # (Mb)随机生成用户数量的任务
        for _ in range(len(task_size)):
            TASK.append(Task(_, task_size=task_size[_]))
        return TASK, task_size

    def task_schedule(self, action, update_queue=True):
        edge_task = []  # LEO处理的任务集合
        for i in range(len(action)):
            self.edge_count += 1
            edge_task.append(self.TASK[i])
        self.edge_computing(edge_task, update_queue)  # LEO处理


    # leo处理
    def edge_computing(self, task, update_queue=True):
        if len(task) == 0:
            return

        mid = (self.N + 1) / 2  # 接入卫星标号

        # 判断用户到达相应卫星的顺序（用来计算排队时延）
        for t in task:
            # 时延

            send_delay = t.task_size / self.send_rate()  # 用户发送时延
            propagate_delay = high_leo / c  # 用户到接入卫星的传播时延
            # 星间链路的转发时延（不包括结果返回）
            forward_delay = abs(mid - t.x) * ((t.task_size / r_leo) + (distance_between_leo / c))
            t.delay['send_delay'] = send_delay
            t.delay['propagate_delay'] = propagate_delay
            t.delay['forward_delay'] = forward_delay

            # 能耗
            send_energy = p_user * send_delay
            t.energy['send_energy'] = send_energy + abs(mid - t.x) * (t.task_size / r_leo) * 0.9

        # 计算每颗卫星上任务的时延
        for i in range(self.N):
            # 处理卸载到每颗卫星上的任务集合
            leo = []
            for t in task:
                if t.x - 1 == i:
                    leo.append(t)

            # 对卫星上的任务按照到达顺序计算
            leo.sort(key=lambda x: x.delay['send_delay'] + x.delay['propagate_delay'] + x.delay['forward_delay'])
            # 排队时延=完成已经存在任务的时间（wait）+某个时隙到达任务的按顺序的排队时延（q_delay）
            wait = (self.leo_queue_size[i] * w) / (f_leo * 1e3)
            q_delay = 0  # 第一个到达的任务不需要额外等待
            for t in leo:
                # 时延
                queue_delay = wait + q_delay  # 任务的排队时延
                calculate_delay = (t.task_size * w) / (f_leo * 1e3)
                # 结果回传时延（不包含结果的发送时延）
                forward_delay = abs(mid - t.x) * (distance_between_leo / c)
                propagate_delay = high_leo / c
                t.delay['queue_delay'] += queue_delay
                t.delay['calculate_delay'] += calculate_delay  # 计算延迟
                t.delay['forward_delay'] += forward_delay
                t.delay['propagate_delay'] += propagate_delay  # +=是因为前面进行过预处理
                t.energy['calculate_energy'] += calculate_delay / 12
                # print('LEO_{}:{}'.format(t.x, t.delay.values()))
                # print('LEO_{}_delay:{},LEO_Energy:{}'.format(t.x, np.sum([d for d in t.delay.values()]),
                #                                           np.sum([e for e in t.energy.values()])))
                q_delay += calculate_delay
                if update_queue:
                    self.leo_queue_size[i] += t.task_size


    def step(self, action):
        self.task_schedule(action)  # 任务调度
        delay, energy, cost = [], [], []
        for t in self.TASK:
            # print(t.x)
            # print(t.delay)
            # print(t.energy)
            # print('='*20)
            d = np.sum([d for d in t.delay.values()])
            d = d / 4.8
            e = np.sum([e for e in t.energy.values()])
            e = e / 0.58
            t.cost = (1 - t.energy_need) * d + t.energy_need * e
            delay.append(d)
            energy.append(e)
            cost.append(t.cost)

        reward = -np.sum(cost)

        # 生成t+1时隙的状态
        # （2）更新t+1时隙的leo队列
        for _ in range(len(self.leo_queue_size)):
            self.leo_queue_size[_] = max(self.leo_queue_size[_] - (f_leo * 1e3 * tau) / w, 0)

        self.leo_forward_leo_queue[0] = max(self.leo_forward_leo_queue[0] - r_leo * tau, 0)
        self.leo_forward_geo_queue[0] = max(self.leo_forward_geo_queue[0] - r_geo * tau, 0)
        # （3）t+1时隙的地面云端跳数
        if self.time_slot % 10 == 0:
            if self.cloud_hop[0] == max_hop:
                self.cloud_hop[0] = 0
            else:
                self.cloud_hop[0] += 1
        # self.space_time_factor = np.random.random(self.N)
        # if self.time_slot % 20 == 0:
        #     self.cloud_hop[0] = max(0, min(max_hop, self.cloud_hop[0] + random.randint(min_change, max_change)))
            # print(self.space_time_factor)
        # （4）生成t+1时隙的任务
        self.TASK, task_size = self.task_generate()

        normal_leo_queue = self.leo_queue_size / max_queue_size

        self.state = np.concatenate([task_size, self.leo_queue_size,
                                     self.leo_forward_geo_queue, self.leo_forward_leo_queue, self.cloud_hop])

        return self.state, reward, np.sum(delay), np.sum(energy)


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        # print(action)
        return action[0]


def update(batch_size, gamma=0.85, soft_tau=1e-2, ):
    state, action, reward, next_state = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value = value_net(state)
    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)

    # Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + gamma * target_value
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())

    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()
    # Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action))
    target_value_func = predicted_new_q_value - log_prob
    value_loss = value_criterion(predicted_value, target_value_func.detach())

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    # Training Policy Function
    policy_loss = (log_prob - predicted_new_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


if __name__ == '__main__':
    I, N, T = 10, 3, 100
    env = ENV(I,N)
    EP_LEN = 100
    NUM_EPOCHS = 10
    state_dim = I + N + 3
    action_dim = I
    hidden_dim = 256
    max_frames  = 40000
    max_steps   = 500
    frame_idx   = 0
    rewards     = []
    batch_size  = 128
    episode = 0
    value_net = ValueNetwork(state_dim, hidden_dim).to(device)
    target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

    soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    value_criterion = nn.MSELoss()
    soft_q_criterion1 = nn.MSELoss()
    soft_q_criterion2 = nn.MSELoss()

    value_lr = 1e-4
    soft_q_lr = 1e-4
    policy_lr = 1e-4

    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
    soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
    soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0
        episode += 1
        for step in range(max_steps):
            if frame_idx > 1000:
                action = policy_net.get_action(state).detach()
                # print(action)
                next_state, reward, delay, energy = env.step(action.numpy())
            else:
                action = np.random.choice(5, I)
                # print(action)
                next_state, reward, delay, energy = env.step(action)

            replay_buffer.push(state, action, reward, next_state)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if len(replay_buffer) > batch_size:
                update(batch_size)

        print("episode:%d------>reward----->%f" % (episode, episode_reward))
        rewards.append(episode_reward)

