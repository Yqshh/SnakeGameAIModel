import os
import random
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from snake import SnakeWorld, Cell, Vector

# Константы
MEMORY_LIMIT = 20_000
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
CHECKPOINT = "checkpoint.pth"

# Опыт (для памяти)
@dataclass
class Experience:
    state: np.ndarray
    action: list[int]
    reward: float
    next_state: np.ndarray
    done: bool

# Модель
class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# Тренер
class Trainer:
    def __init__(self, model: nn.Module, lr: float, gamma: float):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states, dtype=np.float32))
        next_states = torch.tensor(np.array(next_states, dtype=np.float32))
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        preds = self.model(states)
        targets = preds.clone()

        for i in range(len(dones)):
            q_val = rewards[i]
            if not dones[i]:
                q_val += self.gamma * torch.max(self.model(next_states[i]))
            targets[i][torch.argmax(actions[i]).item()] = q_val

        self.optimizer.zero_grad()
        loss = self.loss_fn(preds, targets)
        loss.backward()
        self.optimizer.step()

# Агент
class Learner:
    def __init__(self):
        self.exploration_rate = 80
        self.discount = 0.9
        self.memory = deque(maxlen=MEMORY_LIMIT)
        self.model = QNetwork(11, 3)  # 11 признаков, 3 действия
        self.trainer = Trainer(self.model, lr=LEARNING_RATE, gamma=self.discount)
        self.games_played = 0
        self.best_score = 0
        self.scores = []

    def get_state(self, game: SnakeWorld) -> np.ndarray:
        head = game.head
        left = Cell(head.x - 20, head.y)
        right = Cell(head.x + 20, head.y)
        up = Cell(head.x, head.y - 20)
        down = Cell(head.x, head.y + 20)

        dir_l = game.direction == Vector.LEFT
        dir_r = game.direction == Vector.RIGHT
        dir_u = game.direction == Vector.UP
        dir_d = game.direction == Vector.DOWN

        state = [
            (dir_r and game._collision(right)) or
            (dir_l and game._collision(left)) or
            (dir_u and game._collision(up)) or
            (dir_d and game._collision(down)),

            (dir_u and game._collision(right)) or
            (dir_d and game._collision(left)) or
            (dir_l and game._collision(up)) or
            (dir_r and game._collision(down)),

            (dir_u and game._collision(left)) or
            (dir_d and game._collision(right)) or
            (dir_l and game._collision(down)) or
            (dir_r and game._collision(up)),

            dir_l, dir_r, dir_u, dir_d,

            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,
        ]
        return np.array(state, dtype=int)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def train_batch(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = list(self.memory)
        states, actions, rewards, next_states, dones = zip(*[(e.state, e.action, e.reward, e.next_state, e.done) for e in sample])
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_single(self, state, action, reward, next_state, done):
        self.trainer.train_step([state], [action], [reward], [next_state], [done])

    def choose_action(self, state: np.ndarray) -> list[int]:
        eps = max(5, self.exploration_rate - self.games_played)
        moves = [0, 0, 0]
        if random.random() < eps / 100.0:
            move = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            pred = self.model(state_tensor)
            move = torch.argmax(pred).item()
        moves[move] = 1
        return moves

# Сохранение/загрузка
def save_state(agent: Learner):
    checkpoint = {
        "model": agent.model.state_dict(),
        "optimizer": agent.trainer.optimizer.state_dict(),
        "games": agent.games_played,
        "best": agent.best_score,
        "scores": agent.scores,
        "exploration": agent.exploration_rate,
        "discount": agent.discount,
        "memory": list(agent.memory),
    }
    torch.save(checkpoint, CHECKPOINT)

def load_state(agent: Learner):
    if os.path.exists(CHECKPOINT):
        data = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        agent.model.load_state_dict(data["model"])
        agent.trainer.optimizer.load_state_dict(data["optimizer"])
        agent.games_played = data.get("games", 0)
        agent.best_score = data.get("best", 0)
        agent.scores = data.get("scores", [])
        agent.exploration_rate = data.get("exploration", agent.exploration_rate)
        agent.discount = data.get("discount", agent.discount)
        agent.memory = deque(data.get("memory", []), maxlen=MEMORY_LIMIT)
        print(f"Загружено: игры={agent.games_played}, лучший={agent.best_score}")

# Основной цикл
def train():
    agent = Learner()
    game = SnakeWorld()
    load_state(agent)

    while True:
        state_old = agent.get_state(game)
        action = agent.choose_action(state_old)
        reward, done, score = game.step(action)
        state_new = agent.get_state(game)

        agent.train_single(state_old, action, reward, state_new, done)
        agent.store_experience(state_old, action, reward, state_new, done)

        if done:
            game._restart()
            agent.train_batch()
            agent.games_played += 1
            agent.scores.append(score)
            avg = sum(agent.scores) / len(agent.scores)
            agent.best_score = max(agent.best_score, score)

            save_state(agent)
            print(f"Игра {agent.games_played} | Счёт: {score} | Лучший: {agent.best_score} | Средний: {avg:.2f}")

if __name__ == "__main__":
    train()
