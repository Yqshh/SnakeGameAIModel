import pygame
import random
import math
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)

Point = namedtuple('Point', 'x y')

BLOCK_SIZE = 20
SPEED = 40

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - 2*BLOCK_SIZE, self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')

    def _place_food(self):
        x = random.randrange(0, self.w, BLOCK_SIZE)
        y = random.randrange(0, self.h, BLOCK_SIZE)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == [1, 0, 0]:
            new_dir = clock_wise[idx]
        elif action == [0, 1, 0]:
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)

    def play_step(self, action):
        self.frame_iteration += 1
        dist_before = math.dist((self.head.x, self.head.y), (self.food.x, self.food.y))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0.0
        game_over = False

        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10.0
            return reward, game_over, self.score

        dist_after = math.dist((self.head.x, self.head.y), (self.food.x, self.food.y))
        if dist_after < dist_before:
            reward += 2.0
        else:
            reward -= 1.0
        reward -= 0.1

        if self.head == self.food:
            self.score += 1
            reward = 20.0
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        return reward, game_over, self.score

    def _update_ui(self):
        self.display.fill((0, 0, 0))
        for pt in self.snake:            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(score_text, [0, 0])
        pygame.display.flip()
