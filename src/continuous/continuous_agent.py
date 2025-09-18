import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

# -------------------------
# Detecta GPU automaticamente
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Actor e Critic
# -------------------------
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, buffer_size=1000000, batch_size=256):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((np.array(state, dtype=np.float32),
                            np.array(action, dtype=np.float32),
                            float(reward),
                            np.array(next_state, dtype=np.float32),
                            bool(done)))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

# -------------------------
# TD3 Agent com GPU
# -------------------------
class TD3Agent:
    def __init__(self, state_size, action_size,
                 lr=3e-4, gamma=0.99, tau=0.005,
                 buffer_size=1000000, batch_size=256,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):

        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.learn_step = 0

        # Actor e Critic
        self.actor = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_1 = Critic(state_size, action_size).to(self.device)
        self.critic_2 = Critic(state_size, action_size).to(self.device)
        self.critic_1_target = Critic(state_size, action_size).to(self.device)
        self.critic_2_target = Critic(state_size, action_size).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=lr)

        # Replay
        self.memory = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)

    def act(self, state, noise=0.1):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        self.actor.train()
        action += np.random.normal(0, noise, size=self.action_size)
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) >= self.batch_size:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
            Q1_next = self.critic_1_target(next_states, next_actions)
            Q2_next = self.critic_2_target(next_states, next_actions)
            Q_target = rewards + self.gamma * torch.min(Q1_next, Q2_next) * (1 - dones)

        Q1 = self.critic_1(states, actions)
        Q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(Q1, Q_target) + F.mse_loss(Q2, Q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 1.0)
        self.critic_optimizer.step()

        if self.learn_step % self.policy_delay == 0:
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Atualização suave dos targets
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.learn_step += 1
