import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import pickle
import matplotlib.pyplot as plt
from celeste_env import CelesteEnv
import csv
import pandas as pd

# --- Hyperparameters ---
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 30000
TARGET_UPDATE = 2000
MEMORY_CAPACITY = 50000
MAX_EPISODES = 2000
PROGRESS_FILE = "checkpoints/progress.pkl"
CHECKPOINT_FILE = "checkpoints/checkpoint.pth"
CSV_LOG = "checkpoints/training_log.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
#  DQN + Replay Memory
# ==========================================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(193, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            torch.tensor(state, dtype=torch.float32).to(device),
            torch.tensor(action, dtype=torch.long).unsqueeze(1).to(device),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(next_state, dtype=torch.float32).to(device),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)


# ==========================================================
#  Core training functions
# ==========================================================
def select_action(state, steps_done, policy_net, action_space):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)
    if random.random() < eps_threshold:
        return action_space.sample(), eps_threshold
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_tensor)
        return q_values.argmax(1).item(), eps_threshold


def optimize_model(memory, policy_net, target_net, optimizer, scheduler):
    if len(memory) < BATCH_SIZE:
        return 0.0
    state, action, reward, next_state, done = memory.sample(BATCH_SIZE)
    q_values = policy_net(state).gather(1, action)
    next_q = target_net(next_state).max(1)[0].unsqueeze(1).detach()
    expected_q = reward + (1 - done) * GAMMA * next_q
    loss = nn.functional.mse_loss(q_values, expected_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    return loss.item()


# ==========================================================
#  Logging + Plotting
# ==========================================================
def log_to_csv(episode, reward, loss):
    os.makedirs(os.path.dirname(CSV_LOG), exist_ok=True)
    new = not os.path.exists(CSV_LOG)
    with open(CSV_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["episode", "reward", "loss"])
        writer.writerow([episode, reward, loss])


def plot_progress(csv_path):
    if not os.path.exists(csv_path):
        print("No training log yet.")
        return
    data = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(data["episode"], data["reward"], label="Reward", alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================================================
#  Main
# ==========================================================
if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    env = CelesteEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)
    memory = ReplayMemory(MEMORY_CAPACITY)

    steps_done = 0
    episode_rewards = []

    # --- Load previous progress ---
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "rb") as f:
            episode_rewards = pickle.load(f)
        print(f"📈 Loaded {len(episode_rewards)} past episodes")

    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
        policy_net.load_state_dict(checkpoint["policy"])
        target_net.load_state_dict(checkpoint["target"])
        optimizer.load_state_dict(checkpoint["optim"])
        print("✅ Model checkpoint loaded")

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label="Total Reward")
    ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")

    # --- Training Loop ---
    for episode in range(len(episode_rewards), MAX_EPISODES):
        state, _ = env.reset()
        total_reward = 0.0
        losses = []

        for t in range(1000):
            action, eps = select_action(state, steps_done, policy_net, env.action_space)
            next_state, reward, done, _, _ = env.step(action)
            reward = np.clip(reward, -10, 10)
            memory.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps_done += 1

            loss = optimize_model(memory, policy_net, target_net, optimizer, scheduler)
            if loss:
                losses.append(loss)

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        avg_loss = np.mean(losses) if losses else 0.0
        episode_rewards.append(total_reward)
        print(f"Ep {episode:04d} | Reward: {total_reward:7.2f} | Eps: {eps:.3f} | Loss: {avg_loss:.4f}")

        # --- Save progress + checkpoint ---
        with open(PROGRESS_FILE, "wb") as f:
            pickle.dump(episode_rewards, f)

        torch.save({
            "policy": policy_net.state_dict(),
            "target": target_net.state_dict(),
            "optim": optimizer.state_dict(),
        }, CHECKPOINT_FILE)

        log_to_csv(episode, total_reward, avg_loss)

        # --- Live plot update ---
        line.set_data(range(len(episode_rewards)), episode_rewards)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)

        # --- Periodic checkpoint backup ---
        if episode % 100 == 0 and episode > 0:
            torch.save(policy_net.state_dict(), f"checkpoints/dqn_ep{episode}.pth")

    plt.ioff()
    plt.show()
    env.close()

    # After training, show full progress graph
    plot_progress(CSV_LOG)
