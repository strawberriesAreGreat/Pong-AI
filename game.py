import sys
import pygame
import random 
from pygame.locals import *
from collections import namedtuple
from heuristic import heuristic

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

MAX_GAME_SCORE = 10
GAME_SPEED = 50000

class PongGameAI:
    count = 0

    def __init__(self):
        # Init display
        #self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pretty & Smart Pong")
        #self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        #init scores
        self.points = [0,0]
        self.scores = [0,0]
        #init ball
        self.ball = Ball()
        #init paddle
        self.paddles = [0,0]
        self.paddles[0] = Paddle(20, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2, COLOR_A)
        self.paddles[1] = Paddle(SCREEN_WIDTH - 20 - PADDLE_WIDTH, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2, COLOR_B)
        
        self.frame_iteration = 0

    def play_step(self, moveA, moveB):

        #for event in pygame.event.get(): TODO: umcomment this
            #if event.type == QUIT:
                #pygame.quit()
                #sys.exit()
  
        # player movement
        self.paddles[0].move(moveA)
        self.paddles[1].move(moveB)

        # Check if game over 

        isCollision, scorer, bouncer = self._collision_check(self.ball.rect, self.paddles[0].rect, self.paddles[1].rect)

        if isCollision:
            ball_pos = [ self.ball.rect.x, self.ball.rect.y ]
            ball_vel = [ self.ball.dx, self.ball.dy ]
            paddles = [ self.paddles[0].rect.y, self.paddles[1].rect.y ]
            results = [0,0]
            results = heuristic( scorer, bouncer, paddles, ball_pos, ball_vel, self.count)
            if(results[0] > 0):
                self.points[0] += results[0]
                self.scores[0] += results[0]
            elif(results[1] > 0):
                self.points[1] += results[1]
                self.scores[1] += results[1]
        
            print("player a: ", self.points[0] , "player b: ", self.points[1] )
        else:
            self.points = [0,0]

        # Update
        self.ball.update()
        #self._draw() TODO: umcomment this
        #self.clock.tick(GAME_SPEED) TODO: umcomment this

        return self.points, self._is_end(), self.points


    # Check for collisionss
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
        

    def _collision_check(self, ball_rect, paddle_a_rect, paddle_b_rect):
        isCollision = False
        score = [0, 0]
        bounce = [0, 0]

        # Check for collision with paddle A
        if ball_rect.colliderect(paddle_a_rect):
            self.ball.dx = -self.ball.dx
            isCollision = True
            bounce[0] = 1
            # Ensure the ball does not get stuck in the paddle
            ball_rect.left = paddle_a_rect.right

        # Check for collision with paddle B
        if ball_rect.colliderect(paddle_b_rect):
            self.ball.dx = -self.ball.dx
            isCollision = True
            bounce[1] = 1
            # Ensure the ball does not get stuck in the paddle
            ball_rect.right = paddle_b_rect.left

        # Check for scoring conditions
        if ball_rect.left <= 0:
            isCollision = True
            score[1] = 1  # Player B scores
            self.scores[1] += 1
            self.ball.reset()
            self.paddles[0].reset()
            self.paddles[1].reset()
        elif ball_rect.right >= SCREEN_WIDTH:
            isCollision = True
            score[0] = 1  # Player A scores
            self.scores[0] += 1
            self.ball.reset()
            self.paddles[0].reset()
            self.paddles[1].reset()
        # Ensure the ball does not get stuck in the top or bottom boundaries
        if ball_rect.top <= 0 or ball_rect.bottom >= SCREEN_HEIGHT:
            ball_rect.clamp_ip(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

        return isCollision, score, bounce

        #checking for end of game
    def _is_end(self):
        if self.scores[0] == MAX_GAME_SCORE: 
            return True
        elif self.scores[1] == MAX_GAME_SCORE: 
            return True
        else:
            return False
                
    # Draw game objects
    def _draw(self):
        self.screen.fill(COLOR_BACKGROUND)
        self.screen.blit(self.paddles[0].image, self.paddles[0].rect)
        self.screen.blit(self.paddles[1].image, self.paddles[1].rect)
        self.screen.blit(self.ball.image, self.ball.rect)

        # Display the scores
        score_text = score_font.render(f"{self.scores[0]} - {self.scores[1]}", True, COLOR_TEXT)
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
    def reset(self):
        self.rect.y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2

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
