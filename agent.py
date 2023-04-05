import torch 
import random 
import numpy as np
from collections import deque
from game import PongGameAI, Point 
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
DEVICE = torch.device(dev)  

class Agent: 
    def __init__(self, player, color):
        self.name = color
        self.n_games = 0
        self.epsilon = 0 #rate of randomnes /
        self.gamma = 0.9 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.player = player
        self.enemy = 1 - player
        self.record = 0
        self.model  = Linear_QNet(6, 256, 3)
        self.model.to(DEVICE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):

        paddle = game.paddles[self.player]

        player_paddle_y = paddle.rect.y
        enemy_paddle_y = game.paddles[self.enemy].rect.y 

        state = [ 
            player_paddle_y,
            enemy_paddle_y,
            game.ball.rect.y,
            game.ball.rect.x,
            game.ball.dx,
            game.ball.dy
        ]
        return state 
       
    
    
    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append( (state, action, reward, next_state, game_over) )

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 10 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
            #print("Move ", final_move)
        else:
            # predict move
            state0 = torch.tensor(state, dtype=torch.float).to(DEVICE)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            #print("Move ", final_move)
            final_move[move] = 1

        return final_move

def train():
    plot_scores = [[],[]]
    plot_mean_scores = [[],[]]
    total_score = [0, 0]
    game = PongGameAI()
    agents = [ 
        Agent(0,"purple"), 
        Agent(1, "green") 
    ]
    final_moves = [ [], []]

    while True:

        for i, agent in enumerate(agents):
            # get old state
            state_old = agent.get_state(game)   
            # get move
            final_moves[i] = agent.get_action(state_old)
        
        # perform move and get new state
        rewards, game_over, scores = game.play_step( final_moves[0], final_moves[1] )

        for i, agent in enumerate(agents):
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_moves[i], rewards[i], state_new, game_over)

            # remember
            agent.remember(state_old, final_moves[i], rewards[i], state_new, game_over)

            if game_over:
                # train long memory, plot result
                game.reset()
                
                plots = []

                for i, agent in enumerate(agents):
                    agent.n_games += 1
                    agent.train_long_memory()

                    if scores[i] > agent.record:
                        agent.record = scores[i]
                        agent.model.save( agent.name + '.model')
                    
                    plot_scores[i].append(rewards[i])
                    total_score[i] += scores[i]
                    mean_score = total_score[i] / agents[i].n_games
                    plot_mean_scores[i].append(mean_score)
                    plots.append(plot_scores[i])
                    plots.append(plot_mean_scores[i])
                
                plot(plots[0], plots[1], plots[2], plots[3])

        if game_over:
            game.count += 1
            print('Game', agents[i].n_games, ':\nPurple Score', scores[0], ' Record:', agents[0].record)
            print('Green Score', scores[1], ' Record:', agents[1].record)
                

            
if __name__ == '__main__':
    train()