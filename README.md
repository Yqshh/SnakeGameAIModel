# Snake AI (PyTorch, DQN)

## Описание(description)

Этот проект реализует обучение нейросети играть в классическую игру «Змейка» с помощью алгоритма Deep Q-Learning (DQN).  
Используется библиотека PyTorch для построения и обучения модели, а также pygame для графического интерфейса.

This project implements training a neural network to play the classic Snake game using the Deep Q-Learning (DQN) algorithm.  
It uses PyTorch for building and training the model, and pygame for the graphical interface.

### 🚀 Возможности
- Обучение агента играть в змейку с использованием ε-жадной стратегии.
- Сохранение и загрузка чекпойнтов (весов модели, оптимизатора, статистики).
- GUI-режим для демонстрации игры обученного агента.
- Логирование результатов обучения.

### 🚀 Features
- Train an agent to play Snake using the ε-greedy strategy.
- Save and load checkpoints (model weights, optimizer state, training statistics).
- GUI mode to demonstrate the trained agent.
- Training logs for performance tracking.

### 📂 Структура проекта(Project Structure)
- `snake.py` — логика игры(game logic).
- `model.py` — агент, нейросеть, обучение(agent, neural network, training loop).
- `gui.py` — графический интерфейс для игры с ИИ(graphical interface with AI).
- `checkpoint.pth` — сохранённый прогресс обучения(saved training progress).
- `training_log.csv` — журнал обучения(training log).

### ⚙️ Установка(Installation)
```bash
git clone <репозиторий>
cd snake-ai
pip install -r requirements.txt
