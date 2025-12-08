import os
import pygame
import torch

from snake import SnakeGame, Direction
from model import Agent, CHECKPOINT_FILE

pygame.init()
FONT = pygame.font.SysFont("arial", 25)

class SnakeGUI:
    def __init__(self, ai=True, fps=15, show_model_info=True):
        # Инициализация игры и GUI
        self.game = SnakeGame()
        self.ai = ai
        self.fps = fps
        self.show_model_info = show_model_info

        # Экран и таймер
        self.display = pygame.display.set_mode((self.game.w, self.game.h))
        pygame.display.set_caption("Snake (GUI)")
        self.clock = pygame.time.Clock()

        # Информация о загрузке модели
        self.loaded_info = ""

        # ИИ-агент
        if ai:
            self.agent = Agent()
            # В GUI — детерминированный режим: без ε-рандома
            self.agent.epsilon = 0

            # Пытаемся загрузить чекпойнт
            self._load_checkpoint_into_agent()

    def _load_checkpoint_into_agent(self):
        #Загрузка чекпойнта
        if not os.path.exists(CHECKPOINT_FILE):
            self.loaded_info = "Checkpoint not found"
            print("⚠️ Чекпойнт не найден. Игра будет без ИИ.")
            self.ai = False
            return

        try:
            # В PyTorch 2.6 по умолчанию weights_only=True — нам нужен словарь, поэтому выключаем
            checkpoint = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)

            # Проверка структуры
            if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
                raise RuntimeError("Файл не является чекпойнтом словаря: отсутствует ключ 'model_state'")

            # Загрузка только весов модели для инференса
            self.agent.model.load_state_dict(checkpoint["model_state"])
            self.agent.model.eval()

            best = checkpoint.get("best_score", 0)
            games = checkpoint.get("n_games", 0)
            self.loaded_info = f"Loaded checkpoint: best={best}, games={games}"
            print(f"Загружен чекпойнт: лучший={best}, игр={games}")

        except Exception as e:
            self.loaded_info = f"Load error: {e}"
            print("Ошибка загрузки чекпойнта:", e)
            # Отключаем ИИ, чтобы GUI не падал
            self.ai = False

    def draw(self):
        # Отрисовка змейки, еды, HUD
        self.display.fill((0, 0, 0))

        # Змейка
        for pt in self.game.snake:
            pygame.draw.rect(self.display, (0, 200, 0), pygame.Rect(pt.x, pt.y, 20, 20))
            pygame.draw.rect(self.display, (0, 120, 0), pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # Еда
        pygame.draw.rect(self.display, (220, 20, 60), pygame.Rect(self.game.food.x, self.game.food.y, 20, 20))

        # HUD: счёт
        score_text = FONT.render(f"Score: {self.game.score}", True, (255, 255, 255))
        self.display.blit(score_text, [10, 10])

        # HUD: информация о модели
        if self.show_model_info and self.loaded_info:
            info_text = FONT.render(self.loaded_info, True, (180, 180, 180))
            self.display.blit(info_text, [10, 40])

        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if self.ai:
                # ИИ
                state = self.agent.get_state(self.game)
                action = self._greedy_action(state)
                reward, done, score = self.game.play_step(action)
            else:
                # Ручки
                self.manual_control()
                reward, done, score = self.game.play_step([1, 0, 0])

            # Рендер и тайминг
            self.draw()
            self.clock.tick(self.fps)

            # Завершение эпизода
            if done:
                print(f"Игра окончена. Счёт: {score}")
                pygame.time.delay(250)
                self.game.reset()

    def manual_control(self):
        # Управление направлением стрелками
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.game.direction = Direction.LEFT
        elif keys[pygame.K_RIGHT]:
            self.game.direction = Direction.RIGHT
        elif keys[pygame.K_UP]:
            self.game.direction = Direction.UP
        elif keys[pygame.K_DOWN]:
            self.game.direction = Direction.DOWN

    def _greedy_action(self, state):
        # Чистый greedy-инференс (без случайности)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32)
            q = self.agent.model(state_t)
            move = torch.argmax(q).item()

        if move == 0:
            return [1, 0, 0]  # прямо
        elif move == 1:
            return [0, 1, 0]  # вправо
        else:
            return [0, 0, 1]  # влево

if __name__ == "__main__":
    # ai=True - ИИ; ai=False - ручной режим
    gui = SnakeGUI(ai=True, fps=19, show_model_info=True)
    gui.run()
