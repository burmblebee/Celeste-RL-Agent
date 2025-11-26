# train_dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt
import csv
from collections import deque
from celeste_env import CelesteEnv

# ----------------------------
# Hyperparameters
# ----------------------------
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
EPS_DECAY = 5000
EPS_START = 0.9
EPS_END   = 0.1
TARGET_UPDATE = 1000        # kept as a backup hard-sync
MEMORY_CAPACITY = 200000
MAX_EPISODES = 2000
PROGRESS_FILE = "checkpoints/progress.pkl"
CHECKPOINT_FILE = "checkpoints/checkpoint.pth"
CSV_LOG = "checkpoints/training_log.csv"

# PER hyperparams
PER_ALPHA = 0.6    # how much prioritization is used (0 = uniform)
PER_BETA_START = 0.4
PER_BETA_FRAMES = 200000  # anneal beta to 1.0 over this many steps
EPS_PRIORITY = 1e-6

# training tweaks
ACTION_REPEAT = 3       # sticky actions: repeat each chosen action for N frames
GRAD_CLIP = 0.5
TAU = 0.005             # soft target update factor
Q_REG = 1e-6            # regularization for Q-values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# Compressed action set (curated)
# ==========================================================
# Each action is an array of 5 floats: [moveX, moveY, jump, dash, grab]
ACTIONS = [
    # np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),   # noop
    np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # left
    np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),   # right
    np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),   # jump
    np.array([-1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),  # left + jump
    np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),   # right + jump
    np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),   # dash (dir via moveX/moveY usually)
    np.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),   # dash right
    np.array([-1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),  # dash left
    np.array([0.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float32),  # dash up (moveY negative/up)
    np.array([1.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float32),  # dash up-right
    np.array([-1.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float32), # dash up-left
    np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),   # dash down
     np.array([1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),   # dash down-right
    np.array([-1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),  # dash down-left
    np.array([0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # move up
    np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # move down
    np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),   # jump+dash
    np.array([1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),   # right+jump+dash
    np.array([-1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),  # left+jump+dash
    np.array([0.0, -1.0, 1.0, 1.0, 0.0], dtype=np.float32),  # up+jump+dash
    np.array([1.0, -1.0, 1.0, 1.0, 0.0], dtype=np.float32),  # up-right+jump+dash
    np.array([-1.0, -1.0, 1.0, 1.0, 0.0], dtype=np.float32), # up-left+jump+dash
]
# ACTION_NAMES = [
#     "noop",
#     "left",
#     "right",
#     "jump",
#     "left+jump",
#     "right+jump",
#     "dash",
#     "dash_right",
#     "dash_left",
#     "dash_up",
#     "dash_up-right",
#     "dash_up-left",
#     "dash_down",
#     "dash_down-right",
#     "dash_down-left",
#     "move_up",
#     "move_down",
#     "jump+dash",
#     "right+jump+dash",
#     "left+jump+dash",
#     "up+jump+dash",
#     "up-right+jump+dash",
#     "up-left+jump+dash"
# ]
NUM_ACTIONS = len(ACTIONS)


# ==========================================================
# Dueling network with separate streams
# ==========================================================
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
        )
        # value head
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # advantage head
        self.adv_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.adv_stream(f)
        # combine: Q = V + (A - mean(A))
        return v + a - a.mean(dim=1, keepdim=True)


# ==========================================================
# Lightweight Proportional PER (not super-optimized but simple)
# Stores transitions and priorities in parallel arrays.
# ==========================================================
class PrioritizedReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.size = 0

    def push(self, transition, priority=None):
        # transition: (state, action_idx, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        if priority is None:
            # give new transition maximum priority so it gets sampled at least once
            max_prio = self.priorities.max() if self.size > 0 else 1.0
            self.priorities[self.pos] = max_prio
        else:
            self.priorities[self.pos] = priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, alpha=PER_ALPHA, beta=1.0):
        if self.size == 0:
            raise ValueError("Sampling from empty buffer")

        prios = self.priorities[:self.size] + EPS_PRIORITY
        probs = prios ** alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

        # importance-sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize for stability

        return (
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
            indices,
            torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)
        )

    def update_priorities(self, indices, priorities):
        for idx, pr in zip(indices, priorities):
            self.priorities[idx] = pr

    def __len__(self):
        return self.size


# ==========================================================
# Utilities
# ==========================================================
def soft_update(target, source, tau):
    for tparam, sparam in zip(target.parameters(), source.parameters()):
        tparam.data.copy_(tparam.data * (1.0 - tau) + sparam.data * tau)


# Simple state normalization (env already scales some fields),
# this function ensures dtype and small clipping if needed.
def preprocess_state(state):
    s = np.array(state, dtype=np.float32)
    # clip extreme values for safety
    np.clip(s, -10.0, 10.0, out=s)
    return s


# ==========================================================
# Action selection (epsilon-greedy)
# ==========================================================
def select_action(state, steps_done, policy_net):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)
    if random.random() < eps_threshold:
        idx = random.randrange(NUM_ACTIONS)
        return ACTIONS[idx], eps_threshold, idx
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = policy_net(s)
        idx = q.argmax(dim=1).item()
        return ACTIONS[idx], eps_threshold, idx


# ==========================================================
# Optimization routine (uses PER)
# ==========================================================
def optimize_model(memory: PrioritizedReplay, policy_net, target_net, optimizer, frame_idx):
    if len(memory) < BATCH_SIZE:
        return 0.0, 0.0

    # anneal beta from start to 1.0 over PER_BETA_FRAMES
    beta = min(1.0, PER_BETA_START + (1.0 - PER_BETA_START) * (frame_idx / PER_BETA_FRAMES))

    state, action_idx, reward, next_state, done, indices, weights = memory.sample(BATCH_SIZE, alpha=PER_ALPHA, beta=beta)

    # current Q
    q_values = policy_net(state)  # (B, NUM_ACTIONS)
    q_taken = q_values.gather(1, action_idx)  # (B,1)

    with torch.no_grad():
        # Double DQN: select next action with policy_net, evaluate with target_net
        next_policy_q = policy_net(next_state)
        next_actions = next_policy_q.argmax(dim=1, keepdim=True)
        next_target_q = target_net(next_state).gather(1, next_actions)
        target_q = reward + GAMMA * (1.0 - done) * next_target_q

    # element-wise TD error
    td_errors = (q_taken - target_q).detach().squeeze(1)
    new_priorities = np.abs(td_errors.cpu().numpy()) + EPS_PRIORITY

    # loss with importance-sampling weights and Smooth L1 (Huber)
    losses = nn.functional.smooth_l1_loss(q_taken, target_q, reduction='none')  # (B,1)
    weighted_loss = (losses * weights).mean()

    # q-value regularizer
    q_reg = Q_REG * (q_values.pow(2).mean())
    loss = weighted_loss + q_reg

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP)
    optimizer.step()

    # update priorities in replay
    memory.update_priorities(indices, new_priorities)

    # soft-update target
    soft_update(target_net, policy_net, TAU)

    return loss.item(), td_errors.abs().mean().item()


# ==========================================================
# Logging helper
# ==========================================================
def log_to_csv(episode, reward, loss):
    os.makedirs(os.path.dirname(CSV_LOG), exist_ok=True)
    new = not os.path.exists(CSV_LOG)
    with open(CSV_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["episode", "reward", "loss"])
        writer.writerow([episode, reward, loss])


# ==========================================================
# Main training loop
# ==========================================================
def main():
    os.makedirs("checkpoints", exist_ok=True)

    env = CelesteEnv()
    state, _ = env.reset()
    state_dim = env.observation_space.shape[0]

    policy_net = DuelingDQN(state_dim, NUM_ACTIONS).to(device)
    target_net = DuelingDQN(state_dim, NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = PrioritizedReplay(MEMORY_CAPACITY)

    steps_done = 0
    frame_idx = 0
    episode_rewards = []

    # optionally load progress/checkpoint
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "rb") as f:
            episode_rewards = pickle.load(f)

    if os.path.exists(CHECKPOINT_FILE):
        ckpt = torch.load(CHECKPOINT_FILE, map_location=device)
        policy_net.load_state_dict(ckpt["policy"])
        target_net.load_state_dict(ckpt["target"])
        optimizer.load_state_dict(ckpt["optim"])

    # live plotting
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label="Total Reward")
    ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")

    for episode in range(len(episode_rewards), MAX_EPISODES):
        raw_state, _ = env.reset()
        state = preprocess_state(raw_state)
        total_reward = 0.0
        losses = []

        done = False
        for t in range(10000):  # long horizon per episode
            # choose action
            action_vec, eps, action_idx = select_action(state, steps_done, policy_net)

            # sticky action repeat
            accumulated_reward = 0.0
            next_s = None
            terminated = truncated = False
            for _ in range(ACTION_REPEAT):
                next_raw, r, term, trunc, info = env.step(action_vec)
                accumulated_reward += r
                frame_idx += 1
                steps_done += 1
                if term or trunc:
                    terminated, truncated = term, trunc
                    break
                # if not done, continue repeating

            done_flag = terminated or truncated
            next_s = preprocess_state(next_raw)

            # store transition in PER
            memory.push((state, int(action_idx), float(accumulated_reward), next_s, float(done_flag)))

            state = next_s
            total_reward += accumulated_reward

            # optimize
            if len(memory) >= BATCH_SIZE:
                loss_val, td_avg = optimize_model(memory, policy_net, target_net, optimizer, frame_idx)
                if loss_val:
                    losses.append(loss_val)

            # hard target sync occasionally as backup
            if frame_idx % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done_flag:
                break

        avg_loss = float(np.mean(losses)) if losses else 0.0
        episode_rewards.append(total_reward)

        print(f"Ep {episode:04d} | Reward: {total_reward:7.2f} | Eps: {eps:.3f} | Loss: {avg_loss:.4f}")

        # save progress & checkpoint
        with open(PROGRESS_FILE, "wb") as f:
            pickle.dump(episode_rewards, f)

        torch.save({
            "policy": policy_net.state_dict(),
            "target": target_net.state_dict(),
            "optim": optimizer.state_dict(),
        }, CHECKPOINT_FILE)

        log_to_csv(episode, total_reward, avg_loss)

        # update live plot
        line.set_data(range(len(episode_rewards)), episode_rewards)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)

        if episode % 50 == 0 and episode > 0:
            torch.save(policy_net.state_dict(), f"checkpoints/ddqn_ep{episode}.pth")

    plt.ioff()
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
