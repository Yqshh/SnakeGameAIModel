import pygame
import random
import math
from enum import Enum
from dataclasses import dataclass

pygame.init()
FONT = pygame.font.SysFont("arial", 25)

# Константы
CELL_SIZE = 20
GAME_SPEED = 40
COLOR_BG = (0, 0, 0)
COLOR_SNAKE = (0, 200, 0)
COLOR_FOOD = (200, 0, 0)
COLOR_TEXT = (255, 255, 255)

@dataclass
class Cell:
    x: int
    y: int

class Vector(Enum):
    RIGHT = (1, 0)
    LEFT = (-1, 0)
    UP = (0, -1)
    DOWN = (0, 1)

class SnakeWorld:
    def __init__(self, width: int = 640, height: int = 480):
        self.width, self.height = width, height
        self._restart()

    def _restart(self):
        self.direction = Vector.RIGHT
        center = Cell(self.width // 2, self.height // 2)
        self.body = [
            center,
            Cell(center.x - CELL_SIZE, center.y),
            Cell(center.x - 2 * CELL_SIZE, center.y),
        ]
        self.head = self.body[0]
        self.score = 0
        self.food = None
        self._spawn_food()
        self.steps = 0
        self.surface = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake")

    def _spawn_food(self):
        pos = Cell(
            random.randrange(0, self.width, CELL_SIZE),
            random.randrange(0, self.height, CELL_SIZE),
        )
        if pos in self.body:
            return self._spawn_food()
        self.food = pos

    def _collision(self, point: Cell = None) -> bool:
        point = point or self.head
        return (
            point.x < 0
            or point.x >= self.width
            or point.y < 0
            or point.y >= self.height
            or point in self.body[1:]
        )

    def _advance(self, action: list[int]):
        directions = [Vector.RIGHT, Vector.DOWN, Vector.LEFT, Vector.UP]
        idx = directions.index(self.direction)

        if action == [1, 0, 0]:
            new_dir = directions[idx]
        elif action == [0, 1, 0]:
            new_dir = directions[(idx + 1) % 4]
        else:
            new_dir = directions[(idx - 1) % 4]

        self.direction = new_dir
        dx, dy = self.direction.value
        self.head = Cell(self.head.x + dx * CELL_SIZE, self.head.y + dy * CELL_SIZE)

    def step(self, action: list[int]):
        self.steps += 1
        dist_before = math.dist((self.head.x, self.head.y), (self.food.x, self.food.y))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._advance(action)
        self.body.insert(0, self.head)

        reward, game_over = 0.0, False

        if self._collision() or self.steps > 100 * len(self.body):
            return -10.0, True, self.score

        dist_after = math.dist((self.head.x, self.head.y), (self.food.x, self.food.y))
        reward += 2.0 if dist_after < dist_before else -1.0
        reward -= 0.1

        if self.head == self.food:
            self.score += 1
            reward = 20.0
            self._spawn_food()
        else:
            self.body.pop()

        self._render()
        return reward, game_over, self.score

    def _render(self):
        self.surface.fill(COLOR_BG)
        for segment in self.body:
            pygame.draw.rect(
                self.surface,
                COLOR_SNAKE,
                pygame.Rect(segment.x, segment.y, CELL_SIZE, CELL_SIZE),
            )
        pygame.draw.rect(
            self.surface,
            COLOR_FOOD,
            pygame.Rect(self.food.x, self.food.y, CELL_SIZE, CELL_SIZE),
        )
        score_text = FONT.render(f"Score: {self.score}", True, COLOR_TEXT)
        self.surface.blit(score_text, (0, 0))
        pygame.display.flip()
