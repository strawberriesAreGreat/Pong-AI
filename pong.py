import sys
import pygame
from pygame.locals import *

# Initialize Pygame
pygame.init()
score_font = pygame.font.Font(None, 100)

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
GAME_SPEED = 60


class Paddle(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__() 
        self.image = pygame.Surface((PADDLE_WIDTH, PADDLE_HEIGHT))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def move(self, dy):
        if 0 <= self.rect.y + dy <= SCREEN_HEIGHT - PADDLE_HEIGHT:
            self.rect.y += dy

class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((BALL_SIZE, BALL_SIZE))
        self.image.fill(COLOR_BALL)
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH // 2
        self.rect.y = SCREEN_HEIGHT // 2
        self.dx = 1
        self.dy = 1

    def update(self):
        self.rect.x += self.dx * 5
        self.rect.y += self.dy * 5
        if self.rect.y <= 0 or self.rect.y >= SCREEN_HEIGHT - BALL_SIZE:
            self.dy = -self.dy

    def reset(self):
        self.rect.x = SCREEN_WIDTH // 2
        self.rect.y = SCREEN_HEIGHT // 2
        self.dx = -self.dx

class PongGame:

    def __init__(self):
        # Init display
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pretty & Smart Pong")
        self.clock = pygame.time.Clock()

        #init score
        self.score_a = 0
        self.score_b = 0

        #init ball
        self.ball = Ball()

        #init paddle
        self.paddle_a = Paddle(20, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2, COLOR_A)
        self.paddle_b = Paddle(SCREEN_WIDTH - 20 - PADDLE_WIDTH, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2, COLOR_B)
        
    def play_step(self):

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
  
        keys = pygame.key.get_pressed()
        if keys[K_w]:
            self.paddle_a.move(-5)
        if keys[K_s]:
            self.paddle_a.move(5)
        if keys[K_UP]:
            self.paddle_b.move(-5)
        if keys[K_DOWN]:
            self.paddle_b.move(5)

        # Update
        self._draw()
        self.ball.update()
        self._is_point()
        self.clock.tick(GAME_SPEED)

        return  self._is_end()


        # Check for collisions
    def _is_point(self):
            if self.ball.rect.colliderect(self.paddle_a.rect) or self.ball.rect.colliderect(self.paddle_b.rect):
                self.ball.dx = -self.ball.dx

            if self.ball.rect.x < 0:
                self.score_b += 1
                self.ball.reset()

            if self.ball.rect.x > SCREEN_WIDTH - BALL_SIZE:
                self.score_a += 1
                self.ball.reset()

        #checking for end of game
    def _is_end(self):
            if self.score_a == MAX_GAME_SCORE: 
                print("Game Over")
                return True, "purple", self.score_a, self.score_b

            elif self.score_b == MAX_GAME_SCORE: 

                print("Game Over")
                return True, "green", self.score_b, self.score_a
            
            else:
                return False, 5 , 5, 5
                
    # Draw game objects
    def _draw(self):
        self.screen.fill(COLOR_BACKGROUND)
        self.screen.blit(self.paddle_a.image, self.paddle_a.rect)
        self.screen.blit(self.paddle_b.image, self.paddle_b.rect)
        self.screen.blit(self.ball.image, self.ball.rect)

        # Display the score
        score_text = score_font.render(f"{self.score_a} - {self.score_b}", True, COLOR_TEXT)
        self.screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

        pygame.display.flip()


if __name__ == "__main__":
    game = PongGame()
    while True:
        game_over, winner, score_1, score_2 = game.play_step()

        if game_over: 
            break
    
    print('Winner is player ', winner, ' with score ', score_1)