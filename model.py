import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from collections import deque
from game import PongGameAI, Point

MODEL_FOLDER_PATH = "./models"

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        #self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        #x = F.relu(self.linear2(x))
        x = self.linear3(x)
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
            self.model = model
            self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
            self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
            state = torch.tensor(state, dtype = torch.float)
            next_state = torch.tensor(next_state, dtype = torch.float)
            action = torch.tensor(action, dtype = torch.long)
            reward = torch.tensor(reward, dtype = torch.float)

            if len(state.shape) == 1:
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                reward = torch.unsqueeze(reward, 0)
                action = torch.unsqueeze(action, 0)
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
