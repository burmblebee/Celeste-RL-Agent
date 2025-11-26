from pyexpat import features
from time import time
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



    def __init__(self, host="127.0.0.1", port=5000, live_plot=True, plot_interval=10):
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
                # Player features (8) + exit dx/dy (2) + flattened vision grid
        # features: [posX, posY, velX, velY, dashes, grab, on_ground, facing, exit_dx, exit_dy]
        self.vision_width = 32
        self.vision_height = 32
        vision_size = self.vision_width * self.vision_height

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10 + vision_size,), dtype=np.float32
        )

        # Continuous action vector: [moveX, moveY, jump, dash, grab]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        print("Action space:", self.action_space)
        print("Action space shape:", self.action_space.shape)

        # Always initialize these
        self.last_action = np.zeros(5, dtype=np.float32)
        self.hold_threshold = 0.5  # how strongly the NN has to press to start/keep holding
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
        """
        Send a continuous action vector [moveX, moveY, jump, dash, grab]
        and receive observation + reward.
        """
        # Clip to ensure valid ranges
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Smooth hold behavior for jump/dash/grab
        hold_indices = [2, 3, 4]  # jump, dash, grab

        # Jump/dash are edge-triggered
        for i in hold_indices:
            if action[i] > self.hold_threshold and self.last_action[i] <= self.hold_threshold:
                action[i] = 1.0
            else:
                action[i] = 0.0

        # Remember last action
        self.last_action = action.copy()
        # Build JSON message
        action_msg = json.dumps({"actions": action.tolist()}) + "\n"

        print("Sending action:", action_msg)
        for _ in range(3):
            try:
                self.client.sendall(action_msg.encode("utf-8"))
                break
            except (BrokenPipeError, ConnectionResetError):
                time.sleep(0.01)

        # Receive new observation vector (this also sets self.last_msg to the raw JSON)
        obs = self._get_features()

        # IMPORTANT: compute reward from the raw JSON message (self.last_msg),
        # not from the processed observation vector.
        if not hasattr(self, "last_msg") or not isinstance(self.last_msg, dict):
            # defensive debug output if something went wrong upstream
            print("Warning: last_msg missing or not a dict:", type(getattr(self, "last_msg", None)))
            reward, done = 0.0, False
        else:
            reward, done = self._compute_reward(self.last_msg)

        # optionally update internal state used elsewhere
        self.state = obs

        # live-plot bookkeeping (if you want to track these fields)
        if self.live_plot:
            try:
                # try to collect some history fields if present in last_msg
                p = self.last_msg.get("Player", {})
                self.history["pos_x"].append(p.get("X", 0))
                self.history["pos_y"].append(p.get("Y", 0))
                sp = p.get("Speed", {})
                self.history["vel_x"].append(sp.get("X", 0))
                self.history["vel_y"].append(sp.get("Y", 0))
                self.history["reward"].append(reward)
                if len(self.history["reward"]) % self.plot_interval == 0:
                    self._update_live_plot()
            except Exception:
                pass

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
                    self.last_msg = msg         # <--- store for reward computation
                    return self._to_features(msg)
                except json.JSONDecodeError:
                    continue

    def _to_features(self, msg):
        p = msg["Player"]
        self._save_input_snapshot(msg)

        # Base player features (world units)
        pos_x = float(p["X"])
        pos_y = float(p["Y"])
        vel_x = float(p["Speed"]["X"])
        vel_y = float(p["Speed"]["Y"])
        dashes = float(p.get("Dashes", 0))
        grab = 1.0 if p.get("GrabToggled", False) else 0.0
        on_ground = 1.0 if p.get("OnGround", False) else 0.0
        facing = 1.0 if p.get("Facing", "Right") == "Right" else -1.0

        # Exit: if present, compute vector from player to exit
        exit_dx_world = 0.0
        exit_dy_world = 0.0
        if "Exit" in msg and msg["Exit"] is not None:
            try:
                exit_x = float(msg["Exit"].get("X", pos_x))
                exit_y = float(msg["Exit"].get("Y", pos_y))
                exit_dx_world = exit_x - pos_x
                exit_dy_world = exit_y - pos_y
            except Exception:
                exit_dx_world = 0.0
                exit_dy_world = 0.0

        # Pack features into array (world units for pos and exit; we'll scale below)
        features = np.array([
            pos_x, pos_y,
            vel_x, vel_y,
            dashes,
            grab,
            on_ground,
            facing,
            exit_dx_world,
            exit_dy_world
        ], dtype=np.float32)

        # Vision grid -> flattened numeric array
        grid = msg.get("Grid", [])
        grid_map = {
            "0": 0.0,   # air
            "1": 1.0,   # solid
            "2": 2.0,   # spike
            "9": 0.5    # player marker
        }

        vision = np.array(
            [[grid_map.get(c, 0.0) for c in row] for row in grid],
            dtype=np.float32
        ).flatten()


        # Scale only position/velocity/exit so magnitudes stay reasonable for networks
        features_scaled = self._scale_features(features)

        return np.concatenate([features_scaled, vision])



    
    def _save_input_snapshot(self, msg):
        """Every 20 frames, save the input JSON + readable grid."""
        if not hasattr(self, "_input_counter"):
            self._input_counter = 0
        self._input_counter += 1

        if self._input_counter % 20 != 0:
            return

        # --- Prepare directory ---
        os.makedirs("logs", exist_ok=True)
        fname_base = f"logs/input_{self._input_counter:05d}"

        # --- Save raw input dict ---
        with open(fname_base + ".json", "w", encoding="utf-8") as f:
            json.dump(msg, f, indent=2)
                    

    def _scale_features(self, features):
        # features indices:
        # 0 posX, 1 posY, 2 velX, 3 velY, 4 dashes, 5 grab, 6 on_ground, 7 facing, 8 exit_dx, 9 exit_dy

        features = features.copy()
        features[0] /= 1000.0  # posX scale
        features[1] /= 1000.0  # posY scale
        features[2] /= 100.0   # velX scale
        features[3] /= 100.0   # velY scale

        # exit vector: scale to roughly same units as position (so /1000)
        features[8] /= 1000.0
        features[9] /= 1000.0

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
    def _compute_reward(self, msg):
        reward = 0.0

        # -------------------------------
        # Pull fields from message
        # -------------------------------
        player = msg["Player"]
        pos_x = player["X"]
        pos_y = player["Y"]
        speed_x = player["Speed"]["X"]
        speed_y = player["Speed"]["Y"]
        on_ground = player["OnGround"]

        exit_info = msg["Exit"]      # None or {"X": ..., "Y": ...}
        has_exit = exit_info is not None

        dead = msg.get("Done", False) and msg.get("Reason", "") == "dead"

        # -------------------------------
        # Initialize persistent variables
        # -------------------------------
        if not hasattr(self, "smooth_x"):
            self.smooth_x = pos_x
            self.smooth_y = pos_y
            self.prev_sx = pos_x
            self.prev_sy = pos_y
            self.idle_steps = 0
            self.best_dist = float("inf")

        # -------------------------------
        # Smooth position to remove frame noise
        # -------------------------------
        self.smooth_x = 0.9 * self.smooth_x + 0.1 * pos_x
        self.smooth_y = 0.9 * self.smooth_y + 0.1 * pos_y

        delta_x = self.smooth_x - self.prev_sx
        delta_y = self.smooth_y - self.prev_sy

        self.prev_sx = self.smooth_x
        self.prev_sy = self.smooth_y

        # -------------------------------
        # Survival reward (tiny, steady)
        # -------------------------------
        reward += 0.05

        # -------------------------------
        # Exit-progress reward
        # -------------------------------
        if has_exit:
            exit_x = exit_info["X"]
            exit_y = exit_info["Y"]
            dist = ((pos_x - exit_x)**2 + (pos_y - exit_y)**2)**0.5

            # reward only strictly forward progress
            if dist < self.best_dist:
                reward += 1.0
                self.best_dist = dist

        else:
            # Tiny exploration shaping (biased toward right/up)
            reward += 0.001 * (1 if delta_x > 0 else -1)
            reward += 0.001 * (1 if delta_y < 0 else -1)

        # -------------------------------
        # Stuck penalty
        # -------------------------------
        if abs(delta_x) < 0.1 and abs(delta_y) < 0.1:
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        if self.idle_steps > 20:
            reward -= 0.5

        # -------------------------------
        # Death penalty (if the env reports death)
        # -------------------------------
        if dead:
            reward -= 5.0

        # -------------------------------
        # Reward clamping for stability
        # -------------------------------
        reward = max(-1.0, min(1.0, reward))
        done = dead 
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
