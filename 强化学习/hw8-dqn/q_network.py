import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np



class QNetwork_cartpole(nn.Module):

    def __init__(self, input_dim, output_dim, lr):
        super(QNetwork_cartpole, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)
    
    def inference(self, obs):
        q_value = self.seq(obs)
        return q_value
    
    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




class QNetwork_breakout(nn.Module):

    def __init__(self, input_shape, output_dim, lr=1e-4, device=None):
        """
        input_shape: tuple (C,H,W)
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = output_dim
        self.lr = lr

        C, H, W = input_shape

        # Conv layers (Nature DQN)
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # 自动计算 flatten_dim
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out = self.conv(dummy)
            flatten_dim = conv_out.view(conv_out.size(0), -1).shape[1]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.to(self.device)

    def inference(self, obs):
        """
        obs: (C,H,W) or (B,C,H,W)
        """
        append_dim = False
        if obs.dim() == 3:
            append_dim = True
            obs = obs.unsqueeze(0)  # (1,C,H,W)
        obs = obs.to(self.device)
        conv_out = self.conv(obs)
        flat = conv_out.view(conv_out.size(0), -1)
        q = self.fc(flat)
        if append_dim:
            return q.squeeze(0)
        return q

    def act(self, obs, epsilon=0.0):
        """
        epsilon-greedy action
        obs: (C,H,W)
        """
        single = False
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
            single = True
        obs = obs.to(self.device)
        with torch.no_grad():
            q = self.forward(obs)
            if torch.rand(1).item() < epsilon:
                a = torch.randint(0, self.output_dim, (q.shape[0],), device=self.device)
            else:
                a = q.argmax(dim=1)
        return a[0].item() if single else a.cpu().numpy()

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

