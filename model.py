import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from collections import deque
from game import PongGameAI, Point

MODEL_FOLDER_PATH = "./models"

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
DEVICE = torch.device(dev)  

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size).to(DEVICE)
        self.linear2 = nn.Linear(hidden_size, hidden_size).to(DEVICE)
        self.linear3 = nn.Linear(hidden_size, hidden_size).to(DEVICE)
        self.linear4 = nn.Linear(hidden_size, output_size).to(DEVICE)
        
    def forward(self, x):
        x = F.relu(self.linear1(x)).to(DEVICE)
        x = F.relu(self.linear2(x)).to(DEVICE)
        x = F.relu(self.linear3(x)).to(DEVICE)
        x = self.linear4(x).to(DEVICE)
        return x    
    
    def save (self, file_name):
        if not os.path.exists(MODEL_FOLDER_PATH):
            os.makedirs(MODEL_FOLDER_PATH)
        file_name = os.path.join(MODEL_FOLDER_PATH, file_name)
        torch.save(self.state_dict(), file_name)
    
class QTrainer: 
    def __init__(self,model,lr,gamma):
            self.lr = lr
            self.gamma = gamma
            self.model = model.to(DEVICE)
            self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
            self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
            state = torch.tensor(state, dtype = torch.float).to(DEVICE)
            next_state = torch.tensor(next_state, dtype = torch.float).to(DEVICE)
            action = torch.tensor(action, dtype = torch.long).to(DEVICE)
            reward = torch.tensor(reward, dtype = torch.float).to(DEVICE)

            if len(state.shape) == 1:
                state = torch.unsqueeze(state, 0).to(DEVICE)
                next_state = torch.unsqueeze(next_state, 0).to(DEVICE)
                reward = torch.unsqueeze(reward, 0).to(DEVICE)
                action = torch.unsqueeze(action, 0).to(DEVICE)
                game_over = (game_over, )

            prediction = self.model(state)
            target = prediction.clone()

            for idx in range(len(game_over)):
                Q_new = reward[idx]
                if not game_over[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                target[idx][torch.argmax(action[idx]).item()] = Q_new

            self.optimizer.zero_grad()
            loss = self.criterion(target, prediction)
            loss.backward()

            self.optimizer.step()
