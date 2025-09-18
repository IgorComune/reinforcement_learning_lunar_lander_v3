import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------
# QNetwork: Arquitetura baseada no paper
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        # 3 camadas como no paper, com 128 neurônios nas ocultas
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Linear na saída (sem ativação)
        return x

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, buffer_size=100000, batch_size=64, device='cuda'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.memory.append((np.array(state, dtype=np.float32),
                            int(action),
                            float(reward),
                            np.array(next_state, dtype=np.float32),
                            bool(done)))

    def sample(self):
        batch = random.sample(self.memory, k=self.batch_size)
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

# -------------------------
# DQN Agent Melhorado - mantendo o nome DQNAgent
# -------------------------
class DQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 device='cpu',
                 lr=1e-3,
                 gamma=0.99,
                 buffer_size=100000,
                 batch_size=64,
                 update_every=4,
                 target_update_every=1000,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.996):  # Mudança: decay multiplicativo
        
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        
        # Redes com arquitetura do paper (128 neurônios)
        self.q_local = QNetwork(state_size, action_size, hidden_size=128).to(device)
        self.q_target = QNetwork(state_size, action_size, hidden_size=128).to(device)
        self.q_target.load_state_dict(self.q_local.state_dict())
        
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, device=device)
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Controle de updates
        self.update_every = update_every
        self.target_update_every = target_update_every
        self.t_step = 0
        self.learn_step = 0
        
        # Epsilon-greedy com decay multiplicativo (como no paper)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
    def act(self, state, greedy=False):
        """Ação epsilon-greedy"""
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            state = state.float().unsqueeze(0).to(self.device)
            
        self.q_local.eval()
        with torch.no_grad():
            qvals = self.q_local(state)
        self.q_local.train()
        
        if greedy or random.random() > self.epsilon:
            action = int(torch.argmax(qvals, dim=1).item())
        else:
            action = int(random.randrange(self.action_size))
        return action
    
    def step(self, state, action, reward, next_state, done):
        """Armazena experiência e aprende quando necessário"""
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        
        if self.t_step % self.update_every == 0 and len(self.memory) >= self.batch_size:
            loss = self.learn()
            if self.learn_step % self.target_update_every == 0 and self.learn_step > 0:
                self.q_target.load_state_dict(self.q_local.state_dict())
            return loss
        return None
    
    def learn(self):
        """Otimização com Double DQN"""
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # Double DQN: ação escolhida pela rede local, avaliada pela target
        next_actions = self.q_local(next_states).argmax(dim=1, keepdim=True)
        Q_next = self.q_target(next_states).gather(1, next_actions)
        Q_targets = rewards + (self.gamma * Q_next * (1.0 - dones))
        Q_expected = self.q_local(states).gather(1, actions)
        
        loss = self.criterion(Q_expected, Q_targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        # Clipping de gradiente para estabilidade
        torch.nn.utils.clip_grad_norm_(self.q_local.parameters(), 1.0)
        self.optimizer.step()
        
        self.learn_step += 1
        return loss.item()
    
    def decay_epsilon(self):
        """Decay multiplicativo como no paper"""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_end, self.epsilon)
    
    def save(self, path):
        """Salva apenas a rede local"""
        torch.save(self.q_local.state_dict(), path)
    
    def load(self, path):
        """Carrega modelo e sincroniza target"""
        self.q_local.load_state_dict(torch.load(path, map_location=self.device))
        self.q_target.load_state_dict(self.q_local.state_dict())
        # IMPORTANTE: Definir epsilon baixo para teste
        self.epsilon = 0.01