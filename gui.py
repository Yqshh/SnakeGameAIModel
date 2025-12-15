import os
import pygame
import torch

from snake import SnakeWorld, Vector
from model import Learner, CHECKPOINT, Experience

pygame.init()
FONT = pygame.font.SysFont("arial", 25)

# Цвета
COLOR_BG = (0, 0, 0)
COLOR_SNAKE = (0, 200, 0)
COLOR_SNAKE_INNER = (0, 120, 0)
COLOR_FOOD = (220, 20, 60)
COLOR_TEXT = (255, 255, 255)
COLOR_INFO = (180, 180, 180)

class GameUI:
    def __init__(self, ai: bool = True, fps: int = 15, show_info: bool = True):
        self.game = SnakeWorld()
        self.ai = ai
        self.fps = fps
        self.show_info = show_info

        self.display = pygame.display.set_mode((self.game.width, self.game.height))
        pygame.display.set_caption("Snake (UI)")
        self.clock = pygame.time.Clock()

        self.info_text = ""
        if ai:
            self.agent = Learner()
            self.agent.exploration_rate = 0  # детерминированный режим
            self.load_model()

    def load_model(self):
        if not os.path.exists(CHECKPOINT):
            self.info_text = "Checkpoint not found"
            print("⚠️ Чекпойнт не найден. Игра будет без ИИ.")
            self.ai = False
            return
        try:
            checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
            if "model" not in checkpoint:
                raise RuntimeError("Некорректный файл чекпойнта")
            self.agent.model.load_state_dict(checkpoint["model"])
            self.agent.model.eval()
            best = checkpoint.get("best", 0)
            games = checkpoint.get("games", 0)
            self.info_text = f"Loaded: best={best}, games={games}"
            print(f"Загружен чекпойнт: лучший={best}, игр={games}")
        except Exception as e:
            self.info_text = f"Load error: {e}"
            print("Ошибка загрузки:", e)
            self.ai = False

    def render(self):
        self.display.fill(COLOR_BG)
        for seg in self.game.body:
            pygame.draw.rect(self.display, COLOR_SNAKE, pygame.Rect(seg.x, seg.y, 20, 20))
            pygame.draw.rect(self.display, COLOR_SNAKE_INNER, pygame.Rect(seg.x + 4, seg.y + 4, 12, 12))
        pygame.draw.rect(self.display, COLOR_FOOD, pygame.Rect(self.game.food.x, self.game.food.y, 20, 20))

        score_text = FONT.render(f"Score: {self.game.score}", True, COLOR_TEXT)
        self.display.blit(score_text, (10, 10))

        if self.show_info and self.info_text:
            info = FONT.render(self.info_text, True, COLOR_INFO)
            self.display.blit(info, (10, 40))

        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if self.ai:
                state = self.agent.get_state(self.game)
                action = self.predict_action(state)
                reward, done, score = self.game.step(action)
            else:
                self.handle_input()
                reward, done, score = self.game.step([1, 0, 0])

            self.render()
            self.clock.tick(self.fps)

            if done:
                print(f"Игра окончена. Счёт: {score}")
                pygame.time.delay(250)
                self.game._restart()

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.game.direction = Vector.LEFT
        elif keys[pygame.K_RIGHT]:
            self.game.direction = Vector.RIGHT
        elif keys[pygame.K_UP]:
            self.game.direction = Vector.UP
        elif keys[pygame.K_DOWN]:
            self.game.direction = Vector.DOWN

    def predict_action(self, state):
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32)
            q_vals = self.agent.model(state_t)
            move = torch.argmax(q_vals).item()
        return [1, 0, 0] if move == 0 else [0, 1, 0] if move == 1 else [0, 0, 1]

if __name__ == "__main__":
    ui = GameUI(ai=True, fps=19, show_info=True)
    ui.run()
