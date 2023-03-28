import sys
import pygame
from pygame.locals import *

# Initialize Pygame
pygame.init()
score_font = pygame.font.Font(None, 36)


# reset the game 
# reward for the agent
# play(action) -> direction
# game_iteration
# is_collision 


# Set up game variables
SCREEN_WIDTH, SCREEN_HEIGHT = 1080, 720
COLOR_BACKGROUND = (0, 0, 0)
COLOR_PADDLE = (255, 255, 255)
COLOR_BALL = (255, 255, 255)
PADDLE_WIDTH, PADDLE_HEIGHT = 15, 60
BALL_SIZE = 15


class Paddle(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((PADDLE_WIDTH, PADDLE_HEIGHT))
        self.image.fill(COLOR_PADDLE)
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
        #initializing the game
        self.score_a = 0
        self.score_b = 0
        self.winner = 0
        
        # Create game objects
        self.paddle_a = Paddle(20, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2)
        self.paddle_b = Paddle(SCREEN_WIDTH - 20 - PADDLE_WIDTH, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2)
        self.ball = Ball()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pretty & Smart Pong")
        
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
        self.ball.update()
        self._is_point()
        self._is_end()
        # Draw
        self._draw()
        # Delay
        pygame.time.delay(16)

        return (self.winner > 0), self.winner, self.score_a, self.score_b

        
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
            if self.score_a == 10: 
                self.score_a = 0
                self.score_b = 0
                self.winner = 1
                print("Game Over")

            if self.score_b == 10: 
                self.score_a = 0
                self.score_b = 0
                self.winner = 2
                print("Game Over")
                
    # Draw game objects
    def _draw(self):
        self.screen.fill(COLOR_BACKGROUND)
        self.screen.blit(self.paddle_a.image, self.paddle_a.rect)
        self.screen.blit(self.paddle_b.image, self.paddle_b.rect)
        self.screen.blit(self.ball.image, self.ball.rect)

        # Display the score
        score_text = score_font.render(f"{self.score_a} - {self.score_b}", True, COLOR_PADDLE)
        self.screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

        pygame.display.flip()


if __name__ == "__main__":
    game = PongGame()
    while True:
        game_over, winner, score_a, score_b = game.play_step()

        if game_over: 
            break
    
    print('Winner is player ', winner, ' with score ', score_a)