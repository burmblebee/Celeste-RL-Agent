import gymnasium
import numpy as np
import socket
import json
import matplotlib.pyplot as plt
from gymnasium import spaces
import os
import pickle


class CelesteEnv(gymnasium.Env):
    """
    Celeste Gym environment reading JSON observations from the GymBridge mod.
    The mod now sends:
    {
        "Room": "...",
        "Player": {...},
        "GridOrigin": {...},
        "Grid": ["................", "...#....@.......", ...]
    }
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, host="127.0.0.1", port=5000, live_plot=False, plot_interval=10):
        super().__init__()

        print("🎧 CelesteEnv listening for connection...")
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        print(f"Listening on {host}:{port}... waiting for Celeste mod")

        self.client, addr = self.server.accept()
        print(f"✅ Connected to Celeste mod at {addr}")

        # Grid size from mod (16×9)
        self.vision_width = 16
        self.vision_height = 9
        vision_size = self.vision_width * self.vision_height

        # Player features (8) + flattened vision grid
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8 + vision_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self.buffer = ""
        self.state = None
        self.live_plot = live_plot
        self.plot_interval = plot_interval
        self.history = {"pos_x": [], "pos_y": [], "vel_x": [], "vel_y": [], "reward": []}

        if self.live_plot:
            plt.ion()
            self.fig, self.axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
            self.lines = {
                "pos_x": self.axs[0].plot([], [], label="pos_x")[0],
                "pos_y": self.axs[0].plot([], [], label="pos_y")[0],
                "vel_x": self.axs[1].plot([], [], label="vel_x")[0],
                "vel_y": self.axs[1].plot([], [], label="vel_y")[0],
                "reward": self.axs[2].plot([], [], label="reward")[0],
            }
            for ax in self.axs:
                ax.legend()
            self.axs[0].set_ylabel("Position")
            self.axs[1].set_ylabel("Velocity")
            self.axs[2].set_ylabel("Reward")
            self.axs[2].set_xlabel("Step")
            plt.tight_layout()

    # ==========================================================
    # Gym API
    # ==========================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        print("Waiting for initial observation from Celeste...")
        self.state = self._get_features()
        self._reset_tracking()
        return self.state, {}

    def step(self, action):
        """Send action, receive observation, compute reward."""
        action_msg = json.dumps({"action": int(action)}) + "\n"
        try:
            self.client.sendall(action_msg.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            raise ConnectionError("Lost connection to Celeste mod while sending action.")

        obs = self._get_features()
        reward, done = self._compute_reward(obs)
        self.state = obs

        if self.live_plot and len(self.history["reward"]) % self.plot_interval == 0:
            self._update_live_plot()

        return obs, reward, done, False, {}

    # ==========================================================
    # Data handling
    # ==========================================================
    def _get_features(self):
        """Receive JSON from Celeste mod and convert to numeric features."""
        empty_reads = 0
        while True:
            try:
                data = self.client.recv(4096)
            except ConnectionResetError:
                raise ConnectionError("Connection reset by Celeste mod.")
            if not data:
                empty_reads += 1
                if empty_reads > 3:
                    raise ConnectionError("Lost connection to Celeste mod (no data after retries).")
                continue

            self.buffer += data.decode("utf-8")
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                try:
                    msg = json.loads(line)
                    # ✅ You can log less frequently to avoid flooding:
                    print("Raw message from Celeste:", msg["Player"]["X"], msg["Player"]["Y"])
                    return self._to_features(msg)
                except json.JSONDecodeError:
                    continue

    def _to_features(self, msg):
        p = msg["Player"]
        features = np.array([
                p["X"], p["Y"],
                p["Speed"]["X"], p["Speed"]["Y"],
                p["Dashes"],
                1.0 if p["GrabToggled"] else 0.0,
                1.0 if p["OnGround"] else 0.0,
                1.0 if p["Facing"] == "Right" else -1.0
            ], dtype=np.float32)
        grid = msg["Grid"]
        grid_map = {"#": 1.0, ".": 0.0, "@": 0.5}
        vision = np.array([
            [grid_map.get(c, 0.0) for c in row] for row in grid
        ], dtype=np.float32).flatten()

        features_scaled = self._scale_features(features)
        return np.concatenate([features_scaled, vision])

                    

    def _scale_features(self, features):
        features[0] /= 1000.0  # posX
        features[1] /= 1000.0  # posY
        features[2] /= 100.0   # velX
        features[3] /= 100.0   # velY
        return features

    def _reset_tracking(self):
        self.prev_x = 0
        self.prev_y = 0
        self.prev_vel_x = 0
        self.prev_can_dash = True
        self.idle_steps = 0
        self.x_at_dash_start = None
        for k in self.history:
            self.history[k].clear()

    # ==========================================================
    # Reward
    # ==========================================================
    def _compute_reward(self, features):
        pos_x, pos_y = features[0] * 1000, features[1] * 1000
        vel_x, vel_y = features[2] * 100, features[3] * 100
        can_dash = features[4] > 0.5
        on_ground = features[6] > 0.5
        facing_right = features[7] > 0
        dead = False  # Mod doesn’t send yet — could add later

        if not hasattr(self, "prev_x"):
            self._reset_tracking()

        delta_x = pos_x - self.prev_x
        delta_y = pos_y - self.prev_y
        delta_vx = vel_x - self.prev_vel_x
        reward = 0.1  # survival baseline
        done = False

        reward += delta_x * 2.0
        if not on_ground and vel_x > 0:
            reward += 0.2

        just_jumped = (on_ground is False and self.prev_y - pos_y > 2)
        if just_jumped and vel_x > 0:
            reward += 6.0

        dashed = (self.prev_can_dash and not can_dash) or (delta_vx > 3.0)
        near_apex = abs(vel_y) < 0.5 and not on_ground
        if dashed:
            self.x_at_dash_start = pos_x
        if dashed and near_apex and vel_x > 0:
            reward += 10.0
        if self.x_at_dash_start is not None and not can_dash:
            dash_gain = pos_x - self.x_at_dash_start
            if dash_gain > 0:
                reward += dash_gain * 0.3
            if dash_gain > 30:
                self.x_at_dash_start = None

        if abs(delta_x) < 0.01 and abs(vel_x) < 0.01:
            self.idle_steps += 1
        else:
            self.idle_steps = 0
        if self.idle_steps > 120:
            reward -= 5.0
            done = True

        if dead:
            reward -= 15.0 + (self.prev_x / 20.0)
            done = True

        reward = np.clip(reward, -50, 50)

        self.history["pos_x"].append(pos_x)
        self.history["pos_y"].append(pos_y)
        self.history["vel_x"].append(vel_x)
        self.history["vel_y"].append(vel_y)
        self.history["reward"].append(reward)

        self.prev_x = pos_x
        self.prev_y = pos_y
        self.prev_vel_x = vel_x
        self.prev_can_dash = can_dash

        return reward, done

    # ==========================================================
    # Live plotting
    # ==========================================================
    def _update_live_plot(self):
        steps = range(len(self.history["reward"]))
        self.lines["pos_x"].set_data(steps, self.history["pos_x"])
        self.lines["pos_y"].set_data(steps, self.history["pos_y"])
        self.lines["vel_x"].set_data(steps, self.history["vel_x"])
        self.lines["vel_y"].set_data(steps, self.history["vel_y"])
        self.lines["reward"].set_data(steps, self.history["reward"])
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.001)

    def render(self, mode="human"):
        pass

    def close(self):
        try:
            self.client.close()
        except:
            pass
        try:
            self.server.close()
        except:
            pass
        if self.live_plot:
            plt.ioff()
            plt.show()
