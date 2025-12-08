import os
import csv
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from snake import SnakeGame, Point, Direction

MAX_MEMORY = 20_000
BATCH_SIZE = 1024
LR = 0.001
LOG_FILE = "training_log.csv"
CHECKPOINT_FILE = "checkpoint.pth"

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden)
        self.linear2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state, dtype=np.float32))
        next_state = torch.tensor(np.array(next_state, dtype=np.float32))
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

class Agent:
    def __init__(self):
        self.epsilon = 80
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)  # 11 признаков
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.n_games = 0
        self.best_score = 0
        self.scores = []

    def get_state(self, game: SnakeGame):
        head = game.head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # опасность прямо
            (dir_r and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_l)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d)),

            # опасность справа
            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)) or
            (dir_l and game._is_collision(point_u)) or
            (dir_r and game._is_collision(point_d)),

            # опасность слева
            (dir_u and game._is_collision(point_l)) or
            (dir_d and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_d)) or
            (dir_r and game._is_collision(point_u)),

            # направление
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # положение еды
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        eps = max(5, self.epsilon - self.n_games)
        final_move = [0, 0, 0]
        if random.random() < eps / 100.0:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float32)
            pred = self.model(state0)
            move = torch.argmax(pred).item()
        final_move[move] = 1
        return final_move

def save_checkpoint(agent):
    checkpoint = {
        "model_state": agent.model.state_dict(),
        "optimizer_state": agent.trainer.optimizer.state_dict(),
        "n_games": agent.n_games,
        "best_score": agent.best_score,
        "scores": agent.scores,
        "epsilon": agent.epsilon,
        "gamma": agent.gamma,
        "memory": list(agent.memory),
        "max_memory": agent.memory.maxlen,
    }
    torch.save(checkpoint, CHECKPOINT_FILE)

def load_checkpoint(agent):
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)
        agent.model.load_state_dict(checkpoint["model_state"])
        agent.trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent.n_games = checkpoint.get("n_games", 0)
        agent.best_score = checkpoint.get("best_score", 0)
        agent.scores = checkpoint.get("scores", [])
        agent.epsilon = checkpoint.get("epsilon", agent.epsilon)
        agent.gamma = checkpoint.get("gamma", agent.gamma)
        mem = checkpoint.get("memory", [])
        maxlen = checkpoint.get("max_memory", agent.memory.maxlen)
        agent.memory = deque(mem, maxlen=maxlen)
        print(f"Загружен чекпойнт: игры={agent.n_games}, лучший={agent.best_score}")

def train():
    agent = Agent()
    game = SnakeGame()
    load_checkpoint(agent)

    while True:
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.train_long_memory()
            agent.n_games += 1
            agent.scores.append(score)
            avg_score = sum(agent.scores) / len(agent.scores)
            agent.best_score = max(agent.best_score, score)

            save_checkpoint(agent)
            print(f"Игра {agent.n_games} | Счёт: {score} | Лучший: {agent.best_score} | Средний: {avg_score:.2f}")

if __name__ == "__main__":
    train()
