import os
import yaml
import argparse
from datetime import datetime
import random
import math
import numpy as np

# from sacd.env import make_pytorch_env
from sacd.agent import SacdAgent, SharedSacdAgent


edge_count = 0
geo_count = 0
ground_count = 0


N = 3
max_hop = 5
max_change = 1
min_change = -1
distance_between_leo = 4481  # KM 相邻两颗卫星之间的距离
high_geo = 30000
high_leo = 780  # KM leo卫星的高度
c = 3e5  # KM/s光速
tau = 0.2  # s 时间片大小
p_user = 1  # W  用户发送功率(33dbm)
p_leo = 5
r_leo = 85  # Mbps 卫星发送速率
r_geo = 110
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
f_leo = 6  # GHz leo卫星计算能力
min_task_size = 50
max_task_size = 150
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
        self.leo_forward_leo_queue = np.zeros(N)  # 初始化leo间队列
        self.leo_forward_geo_queue = np.zeros(N)  # 初始化leo间队列
        self.state = np.zeros(N + 2 * I + 1 + 1)  # 初始化环境状态
        self.time_slot = 1  # 时间片计数器
        self.cloud_hop = random.randint(0, max_hop)
        self.edge_count = edge_count
        self.geo_count = geo_count
        self.ground_count = ground_count

    def reset(self):
        self.__init__(self.I, self.N)
        self.TASK, task_size = self.task_generate()  # 随机生成任务
        self.cloud_hop = random.randint(0, max_hop)
        # self.space_time_factor = np.random.random(self.N)  # 随机初始化时空因子
        self.state = np.concatenate([task_size, self.leo_queue_size,
                                     self.leo_forward_geo_queue, self.leo_forward_leo_queue])
        self.time_slot = 1
        return self.state

    def send_rate(self):
        # print(h)
        return B * math.log2(1 + (p_user * h) / (N0 * self.I * p_user * h))

    def task_generate(self):
        self.time_slot += 1
        TASK = []
        task_size = np.random.randint(min_task_size, max_task_size, self.I) / 1e2  # (Mb)随机生成用户数量的任务
        for _ in range(len(task_size)):
            TASK.append(Task(_, task_size=task_size[_]))
        return TASK, task_size

    def task_schedule(self, action, update_queue=True):
        edge_task = []  # LEO处理的任务集合
        geo_task = []
        ground_task = []
        for i in range(len(action)):
            self.TASK[i].x = action[i]
            if self.TASK[i].x == 0:
                ground_task.append(self.TASK[i])
            elif self.TASK[i].x == N + 1:
                geo_task.append(self.TASK[i])
            else:  # LEO处理
                edge_task.append(self.TASK[i])

        self.edge_computing(edge_task, update_queue)  # LEO处理
        self.geo_computing(geo_task, update_queue)  # LEO处理
        self.ground_computing(ground_task, update_queue)  # LEO处理

    # 本地处理
    # def local_computing(self, task, update_queue=True):
    #     if len(task) == 0:
    #         return
    #     for t in task:
    #         # 本地计算时延
    #         calculate_delay = ((t.task_size * w) / (f_user * 1e3))
    #         queue_delay = (self.users_queue_size[t.index] * w) / (f_user * 1e3)
    #         t.delay['calculate_delay'] = calculate_delay
    #         t.delay['queue_delay'] = queue_delay
    #
    #         # 本地计算能耗
    #         calculate_energy = epsilon * pow(f_user * 1e9, 3) * calculate_delay
    #         t.energy['calculate_energy'] = calculate_energy
    #
    #         # 更新本地队列
    #         if update_queue:
    #             self.users_queue_size[t.index] += t.task_size

    # leo处理
    def edge_computing(self, task, update_queue=True):
        if len(task) == 0:
            return

        mid = (self.N + 1) / 2  # 接入卫星标号

        # 判断用户到达相应卫星的顺序（用来计算排队时延）
        for t in task:
            # 时延
            self.edge_count += 1
            send_delay = t.task_size / self.send_rate()  # 用户发送时延
            propagate_delay = high_leo / c  # 用户到接入卫星的传播时延
            # 星间链路的转发时延（不包括结果返回）
            forward_delay = abs(mid - t.x) * ((t.task_size / r_leo) + (distance_between_leo / c))
            t.delay['send_delay'] = send_delay
            t.delay['propagate_delay'] = propagate_delay
            t.delay['forward_delay'] = forward_delay

            # 能耗
            send_energy = p_user * send_delay
            t.energy['send_energy'] = send_energy

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
                # print('LEO:{}'.format(t.delay.values()))
                q_delay += calculate_delay
                if update_queue:
                    self.leo_queue_size[i] += t.task_size

        # geo处理

    def geo_computing(self, task, update_queue=True):
        if len(task) == 0:
            return

        mid = (self.N + 1) / 2  # 接入卫星标号

        # 判断用户到达相应卫星的顺序（用来计算排队时延）
        for t in task:
            # 时延
            self.geo_count += 1
            send_delay = t.task_size / self.send_rate()  # 用户发送时延
            propagate_delay = high_leo / c  # 用户到接入卫星的传播时延
            # 星间链路的转发时延（不包括结果返回）
            forward_delay = abs(mid - mid) * ((t.task_size / r_leo) + (distance_between_leo / c))
            t.delay['send_delay'] = send_delay
            t.delay['propagate_delay'] = propagate_delay
            t.delay['forward_delay'] = forward_delay

            # 能耗
            send_energy = p_user * send_delay
            t.energy['send_energy'] = send_energy

        # 计算每颗卫星上任务的时延
        # 处理卸载到每颗卫星上的任务集合
        leo = []
        for t in task:
            leo.append(t)

        # 对卫星上的任务按照到达顺序计算
        leo.sort(key=lambda x: x.delay['send_delay'] + x.delay['propagate_delay'] + x.delay['forward_delay'])
        # 排队时延=完成已经存在任务的时间（wait）+某个时隙到达任务的按顺序的排队时延（q_delay）
        wait = self.leo_forward_geo_queue[0] / r_geo
        q_delay = 0  # 第一个到达的任务不需要额外等待
        for t in leo:
            # 时延
            queue_delay = wait + q_delay  # 任务的排队时延
            # 结果回传时延（不包含结果的发送时延）
            forward_delay = t.task_size / r_geo
            propagate_delay = high_leo / c + 2 * high_geo / c
            t.delay['queue_delay'] += queue_delay
            t.delay['forward_delay'] += forward_delay
            t.delay['propagate_delay'] += propagate_delay  # +=是因为前面进行过预处理
            t.energy['send_energy'] += forward_delay * p_leo
            # print('GEO:{}'.format(t.delay.values()))
            q_delay += forward_delay
            if update_queue:
                self.leo_forward_geo_queue[0] += t.task_size

    # 地面云处理
    def ground_computing(self, task, update_queue=True):
        if len(task) == 0:
            return

        # 判断用户到达相应卫星的顺序（用来计算排队时延）
        for t in task:
            # 时延
            self.ground_count += 1
            send_delay = t.task_size / self.send_rate()  # 用户发送时延
            propagate_delay = high_leo / c  # 用户到接入卫星的传播时延
            # 星间链路的转发时延（不包括结果返回）
            forward_delay = self.cloud_hop * ((t.task_size / r_leo) + 2 * (distance_between_leo / c))
            t.delay['send_delay'] = send_delay
            t.delay['propagate_delay'] = propagate_delay
            t.delay['forward_delay'] = forward_delay

            # 能耗
            send_energy = p_user * send_delay
            t.energy['send_energy'] = send_energy

        # 计算每颗卫星上任务的时延
        leo = []
        for t in task:
            leo.append(t)

        # 对卫星上的任务按照到达顺序计算
        leo.sort(key=lambda x: x.delay['send_delay'] + x.delay['propagate_delay'])
        # 排队时延=完成已经存在任务的时间（wait）+某个时隙到达任务的按顺序的排队时延（q_delay）
        wait = self.leo_forward_leo_queue[0] / r_leo
        q_delay = 0  # 第一个到达的任务不需要额外等待
        for t in leo:
            # 时延
            queue_delay = wait + q_delay  # 任务的排队时延
            # 结果回传时延（不包含结果的发送时延）
            forward_delay = t.task_size / r_leo
            propagate_delay = high_leo / c
            t.delay['queue_delay'] += queue_delay
            t.delay['propagate_delay'] += 3 * propagate_delay  # +=是因为前面进行过预处理
            # print('Ground:{}'.format(t.delay.values()))
            q_delay += forward_delay
            if update_queue:
                self.leo_forward_leo_queue[0] += t.task_size

    def step(self, action):
        self.task_schedule(action)  # 任务调度
        delay, energy, cost = [], [], []
        for t in self.TASK:
            # print(t.x)
            # print(t.delay)
            # print(t.energy)
            # print('='*20)
            d = np.sum([d for d in t.delay.values()])
            e = np.sum([e for e in t.energy.values()])
            t.cost = (1 - t.energy_need) * d + t.energy_need * e
            delay.append(d)
            energy.append(e)
            cost.append(t.cost)

        reward = -np.sum(cost)

        # 生成t+1时隙的状态
        # （2）更新t+1时隙的leo队列
        for _ in range(len(self.leo_queue_size)):
            self.leo_queue_size[_] = max(self.leo_queue_size[_] - (f_leo * 1e3 * tau) / w, 0)
        for _ in range(N):
            self.leo_forward_leo_queue[_] = max(self.leo_forward_leo_queue[_] - r_leo * tau, 0)
            self.leo_forward_geo_queue[_] = max(self.leo_forward_geo_queue[_] - r_geo * tau, 0)
        # （3）t+1时隙的地面云端跳数
        # self.space_time_factor = np.random.random(self.N)
        if self.time_slot % 20 == 0:
            self.cloud_hop = max(0, min(max_hop, self.cloud_hop + random.randint(min_change, max_change)))
            # print(self.space_time_factor)
        # （4）生成t+1时隙的任务
        self.TASK, task_size = self.task_generate()

        normal_leo_queue = self.leo_queue_size / max_queue_size

        self.state = np.concatenate([task_size, self.leo_queue_size,
                                     self.leo_forward_geo_queue, self.leo_forward_leo_queue])

        return self.state, reward, np.sum(delay), np.sum(energy)

    # def calculate_Q(self, action):
    #     tasks_size = np.array(self.state[:self.I])
    #     energy_need = np.array(self.state[:self.I])
    #     cost_factor = np.array(self.state[-self.N:])
    #     T = []
    #     local_task, edge_task = [], []
    #     for _ in range(self.I):
    #         t = Task(_, tasks_size[_], energy_need[_])
    #         t.x = action[_]
    #         T.append(t)
    #         if t.x == 0:  # 本地处理
    #             local_task.append(t)
    #         else:  # 边缘计算
    #             edge_task.append(t)
    #
    #     self.local_computing(local_task, update_queue=False)
    #     self.edge_computing(edge_task, update_queue=False)
    #
    #     # self.task_schedule(action,update_queue=False)  # 用了会导致计算多遍时延和能耗
    #     delay, energy, cost = [], [], []
    #     for t in T:
    #         d = np.sum([d for d in t.delay.values()])
    #         e = np.sum([e for e in t.energy.values()])
    #         if t.x == 0:
    #             t.cost = lam * d + (1 - lam) * e
    #         else:
    #             t.cost = lam * d + (1 - lam) * e + cost_multiplier * cost_factor[int(t.x) - 1]
    #         cost.append(t.cost)
    #         delay.append(d)
    #         energy.append(e)
    #     reward = -np.sum(cost)
    #
    #     return reward, np.sum(delay), np.sum(energy)

    def ppo_step(self, action):
        # cost_factor = np.array(self.state[-self.N:])
        self.task_schedule(action)
        delay, energy, cost = [], [], []
        for t in self.TASK:
            d = np.sum([d for d in t.delay.values()])
            e = np.sum([e for e in t.energy.values()])
            t.cost = (1 - t.energy_need) * d + t.energy_need * e
            # + cost_multiplier * cost_factor[int(t.x) - 1]
            delay.append(d)
            energy.append(e)
            cost.append(t.cost)

        reward = -np.sum(cost)

        # 生成t+1时隙的状态
        # （2）更新t+1时隙的leo队列
        for _ in range(len(self.leo_queue_size)):
            self.leo_queue_size[_] = max(self.leo_queue_size[_] - (f_leo * 1e3 * tau) / w, 0)  # 单个时隙能够计算的任务

        for _ in range(N):
            self.leo_forward_leo_queue[_] = max(self.leo_forward_leo_queue[_] - r_leo * tau, 0)
            self.leo_forward_geo_queue[_] = max(self.leo_forward_geo_queue[_] - r_geo * tau, 0)
        # （3）t+1时隙的时空因子(地面云端跳数)
        # self.space_time_factor = np.around(np.random.rand(self.N),decimals=1)
        # # if self.time_slot % 50 ==0:
        # #     # self.location += 1
        # #     # self.space_time_factor = fixed_cost[self.location-1:self.location+2]
        # #     self.space_time_factor = np.around(np.random.rand(3),decimals=1)
        # normal_leo_queue = self.leo_queue_size / max_queue_size
        # cost_factor = normal_leo_queue + self.space_time_factor
        if self.time_slot % 20 == 0:
            self.cloud_hop = max(0, min(max_hop, self.cloud_hop + random.randint(min_change, max_change)))
        # （4）生成t+1时隙的任务
        self.TASK, task_size = self.task_generate()
        self.state = np.concatenate([task_size, self.leo_queue_size,
                                     self.leo_forward_geo_queue, self.leo_forward_leo_queue])

        return self.state, reward, np.sum(delay), np.sum(energy)

def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = ENV(10, 3)
    # test_env = make_pytorch_env(
        # args.env_id, episode_life=False, clip_rewards=False)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent.
    Agent = SacdAgent if not args.shared else SharedSacdAgent
    agent = Agent(
        env=env, log_dir=log_dir, cuda=args.cuda,
        seed=args.seed, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'sacd.yaml'))
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
