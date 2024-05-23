import matplotlib.pyplot as plt

from IPython import display

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
real_rewards = []
#准备环境
seed = 543
def fix(env, seed):
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
import gym
import random
env = gym.make('LunarLander-v2' ,render_mode='rgb_array')
fix(env, seed) # fix the environment Do not revise this !!!
random_action = env.action_space.sample()

env.reset()

# img = plt.imshow(env.render())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCriticDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCriticDiscrete, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
                )

        # critic
        self.value_layer = nn.Sequential(
               nn.Linear(state_dim, 128),
               nn.ReLU(),
               nn.Linear(128, 64),
               nn.ReLU(),
               nn.Linear(64, 1)
               )

    def act(self, state, memory):
        state = torch.from_numpy(state).float()
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.timestep = 0
        self.memory = Memory()

        self.policy = ActorCriticDiscrete(state_dim, action_dim, n_latent_var)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCriticDiscrete(state_dim, action_dim, n_latent_var)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(self.memory.states).detach()
        old_actions = torch.stack(self.memory.actions).detach()
        old_logprobs = torch.stack(self.memory.logprobs).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values: 重用旧样本进行新策略的训练
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 计算Alpha散度：
            alpha = 0.5  # 你可以根据需要调整Alpha的值
            alpha_divergence = alpha * torch.sum((torch.exp(old_logprobs) - torch.exp(logprobs)) / alpha)

            # 计算Surrogate Loss with Alpha divergence penalty:
            surrogate_loss = -(torch.exp(logprobs - old_logprobs.detach()) * alpha_divergence)
            loss = surrogate_loss + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy + alpha_divergence

            # 执行梯度下降步骤
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 将新权重复制到旧策略中：
        self.policy_old.load_state_dict(self.policy.state_dict())

    def step(self, reward, done):
        self.timestep += 1
        # Saving reward and is_terminal:
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

        # update if its time
        if self.timestep % update_timestep == 0:
            self.update()
            self.memory.clear_memory()
            self.timstamp = 0

    def act(self, state):
        return self.policy_old.act(state, self.memory)


state_dim = 8 ### 游戏的状态是个8维向量
action_dim = 4 ### 游戏的输出有4个取值
n_latent_var = 256           # 神经元个数
update_timestep = 1200      # 每多少补跟新策略
lr = 0.002                  # learning rate
betas = (0.9, 0.999)
gamma = 0.99                # discount factor
K_epochs = 4                # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO  论文中表明0.2效果不错
random_seed = 1

agent = PPOAgent(state_dim ,action_dim,n_latent_var,lr,betas,gamma,K_epochs,eps_clip)
# agent.network.train()  # Switch network into training mode
#调参数调参数鸡你太美~
EPISODE_PER_BATCH = 5  # update the  agent every 5 episode
NUM_BATCH = 1000000    # totally update the agent for 300 time


avg_total_rewards, avg_final_rewards = [], []
final = np.zeros((NUM_BATCH, 1))

# prg_bar = tqdm(range(NUM_BATCH))
for i in range(NUM_BATCH):

    log_probs, rewards = [], []
    total_rewards, final_rewards = [], []
    values    = []
    masks     = []
    entropy = 0
    # collect trajectory
    for episode in range(EPISODE_PER_BATCH):
        ### 重开一把游戏
        state = env.reset()[0]
        total_reward, total_step = 0, 0
        seq_rewards = []
        for i in range(1000):  ###游戏未结束

            action = agent.act(state) ### 按照策略网络输出的概率随机采样一个动作
            next_state, reward, done, _, _ = env.step(action) ### 与环境state进行交互，输出reward 和 环境next_state
            state = next_state
            total_reward += reward
            total_step += 1
            rewards.append(reward) ### 记录每一个动作的reward
            agent.step(reward, done)
            if done:  ###游戏结束
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                break

    print(f"rewards: ", len(rewards))  # 这里是重要的这里是重要的这里是重要的这里是重要的这里是重要的这里是重要的这里是重要的这里是重要的这里是重要的这里是重要的
    temp_rewards = len(rewards)
    # print(temp_rewards)
    real_rewards.append(temp_rewards)

    if len(final_rewards)> 0 and len(total_rewards) > 0:
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)


fix(env, seed)
agent.policy.eval() # set the network into evaluation mode
test_total_reward = []
for i in range(5):
    actions = []
    state = env.reset()[0]
    # img = plt.imshow(env.render())
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        actions.append(action)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        #img.set_data(env.render())
        #display.display(plt.gcf())
        #display.clear_output(wait=True)
    test_total_reward.append(total_reward)
#test_total_reward
np.savetxt('LunarLander-Hellinger.csv',real_rewards , delimiter = ',')

# plt.plot(real_rewards)
# plt.title('return')
# plt.show()