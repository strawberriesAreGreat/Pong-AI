import sys
import pygame
import random 
from pygame.locals import *
from collections import namedtuple

# Initialize Pygame
pygame.init()
score_font = pygame.font.Font(None, 100)

Point = namedtuple('Point', 'x, y')

# Set up game variables
COLOR_BACKGROUND = (25, 42, 99)
COLOR_BALL = (204, 255, 232)
COLOR_A = (187,160,202)
COLOR_B = (153,247,171)
COLOR_TEXT = (234,255,168)

SCREEN_WIDTH, SCREEN_HEIGHT = 1080, 720
PADDLE_WIDTH, PADDLE_HEIGHT = 20, 100
BALL_SIZE = 20

MAX_GAME_SCORE = 3
GAME_SPEED = 200

class PongGameAI:

    def __init__(self):
        # Init display
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pretty & Smart Pong")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        #init score
        self.score_a = 0
        self.score_b = 0
        self.points = [0,0]
        #init ball
        self.ball = Ball()
        #init paddle
        self.paddles = [0,0]
        self.paddles[0] = Paddle(20, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2, COLOR_A)
        self.paddles[1] = Paddle(SCREEN_WIDTH - 20 - PADDLE_WIDTH, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2, COLOR_B)
        
        self.frame_iteration = 0

    def play_step(self, moveA, moveB):

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
  
        # player movement
        self.paddles[0].move(moveA)
        self.paddles[1].move(moveB)

        # Check if game over 
        rewards = [0,0]; 

        isCollision, scorer, scoree, bouncer = self.is_collision()
        if isCollision:
            if(scorer != scoree): # TODO: this is messing up the heuristic. Score is random at start change to paddle contacting ball
                rewards[scorer] = 1
                rewards[scoree] = -1
            else:
                rewards[bouncer] = 5  
        else:
            rewards = [0,0]
        
        self.points[0] += rewards[0]
        self.points[1] += rewards[1]


        # Update
        self.ball.update()
        self._draw()
 
        self.clock.tick(GAME_SPEED)

        return  self._is_end(rewards) 


    # Check for collisions
    def is_bottom(self, paddle):
        if  paddle.rect.y < 0:
            return 1
        else:
            return 0
        
    def is_top(self, paddle):
        if  paddle.rect.y > SCREEN_HEIGHT - PADDLE_HEIGHT:
            return 1
        else:
            return 0


    def is_collision(self):
            if self.ball.rect.colliderect(self.paddles[0].rect):
                self.ball.dx = -self.ball.dx
                #self.ball.dy += self.paddles[0].direction * 0.5
                return True, 0, 0, 0
            elif self.ball.rect.colliderect(self.paddles[1].rect):
                self.ball.dx = -self.ball.dx
                #self.ball.dy += self.paddles[1].direction * 0.5
                return True, 0, 0, 1

            elif self.ball.rect.x < 0:
                self.score_b += 1
                self.ball.reset()
                return True, 1, 0, 0
            elif self.ball.rect.x > SCREEN_WIDTH - BALL_SIZE:
                self.score_a += 1
                self.ball.reset()
                return True, 0, 1, 0
        
            else: 
                return False, 0, 0, 0

        #checking for end of game
    def _is_end(self, rewards):
            if self.score_a == MAX_GAME_SCORE: 
                return rewards, True, [1,0], [self.points[0], self.points[1]]

            elif self.score_b == MAX_GAME_SCORE: 
                return rewards, True, [0,1], [self.points[0], self.points[1]]
            
            else:
                return rewards, False, [0,0], [self.points[0], self.points[1]]
                
    # Draw game objects
    def _draw(self):
        self.screen.fill(COLOR_BACKGROUND)
        self.screen.blit(self.paddles[0].image, self.paddles[0].rect)
        self.screen.blit(self.paddles[1].image, self.paddles[1].rect)
        self.screen.blit(self.ball.image, self.ball.rect)

        # Display the score
        score_text = score_font.render(f"{self.score_a} - {self.score_b}", True, COLOR_TEXT)
        self.screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

        pygame.display.flip()



class Paddle(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__() 
        self.image = pygame.Surface((PADDLE_WIDTH, PADDLE_HEIGHT))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.direction = 0 # 0 = stay, 1 = up, -1 = down

    def move(self, moves): # TODO: moves only need 2 values not 3
        dy = 0
        if moves[0] > 0: 
            dy = 1 # move up 
            #print("move up")
        elif moves[1] > 0:
            dy = -1 # move down
            #print("move down")

        if 0 <= self.rect.y + dy <= SCREEN_HEIGHT - PADDLE_HEIGHT:
            self.rect.y += dy
            self.direction = -1 if dy < 0 else 1
        else:
            self.direction = 0

class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((BALL_SIZE, BALL_SIZE))
        self.image.fill(COLOR_BALL)
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH // 2
        self.rect.y = SCREEN_HEIGHT // 2
       
        x = random.randint(0,1)
        y = random.randint(0,1)
        self.dx = -1 if x == 0 else 1
        self.dy = -1 if y == 0 else 1


    def update(self):
        self.rect.x += self.dx * 5
        self.rect.y += self.dy * 5
        if self.rect.y <= 0 or self.rect.y >= SCREEN_HEIGHT - BALL_SIZE:
            self.dy = -self.dy

    def reset(self):
        self.rect.x = SCREEN_WIDTH // 2
        self.rect.y = SCREEN_HEIGHT // 2
        
        x = random.randint(0,1)
        y = random.randint(0,1)
        self.dx = -1 if x == 0 else 1
        self.dy = -1 if y == 0 else 1
