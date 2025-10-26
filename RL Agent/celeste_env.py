import gymnasium
import numpy as np
import socket
import json
from gymnasium import spaces

class CelesteEnv(gymnasium.Env):
    """
    OpenAI Gym interface for Celeste.
    - Acts as a TCP server that Celeste (C# mod) connects to.
    - Handles reward, scaling, and step logic.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, host="127.0.0.1", port=5000):
        super(CelesteEnv, self).__init__()

        print("🎧 CelesteEnv listening for connection...")
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        print(f"Listening on {host}:{port}... waiting for Celeste mod")

        self.client, addr = self.server.accept()
        print(f"✅ Connected to Celeste mod at {addr}")


        self.buffer = ""

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(36,), dtype=np.float32
        )


        self.state = None




    def reset(self, *, seed=None, options=None):
        """Reset the environment and return the initial observation + info dict."""
        super().reset(seed=seed)

        print("Waiting for initial features from Celeste...")
        self.state = self._get_features()

        # Gymnasium requires returning (obs, info)
        return self.state, {}


    def step(self, action):
        """
        Take one RL step:
        - Send action to Celeste
        - Receive next state
        - Compute reward
        """
        # ==========================================================
        # ✅ Send action to Celeste mod
        # ==========================================================
        action_msg = json.dumps({"action": int(action)}) + "\n"
        try:
            self.client.sendall(action_msg.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            raise ConnectionError("Lost connection to Celeste mod while sending action.")

        # ==========================================================
        # ✅ Receive next features from Celeste mod
        # ==========================================================
        obs = self._get_features()

        # ==========================================================
        # ✅ Compute reward & termination
        # ==========================================================
        reward, done = self._compute_reward(obs)

        self.state = obs
        return obs, reward, done, False, {}


    def _get_features(self):
        """Receive and parse a JSON line of features from Celeste mod."""
        while True:
            data = self.client.recv(1024).decode("utf-8")
            if not data:
                raise ConnectionError("Lost connection to Celeste mod (no data).")

            self.buffer += data
            if "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                try:
                    msg = json.loads(line)
                    features = np.array(msg["features"], dtype=np.float32)
                    return self._scale_features(features)
                except json.JSONDecodeError:
                    continue


    def _scale_features(self, features):
        """Normalize key numerical values for stability."""
        if len(features) >= 9:
            features[0] /= 1000.0  # posX
            features[1] /= 1000.0  # posY
            features[2] /= 100.0   # velX
            features[3] /= 100.0   # velY
            features[8] /= 110.0   # stamina (max ~110)
        return features


    def _compute_reward(self, features):
        """
        Aggressive reward shaping for early progress in Celeste:
        - Strong positive reward for moving right (toward exit)
        - Extra bonus for upward movement (jumping)
        - Death penalty
        - Small time penalty
        """
        pos_x, pos_y = features[0] * 1000, features[1] * 1000
        vel_x, vel_y = features[2] * 100, features[3] * 100
        dead = features[7] > 0.5
        exit_x, exit_y = features[9], features[10]

        # Horizontal progress (most important)
        if not hasattr(self, "prev_x"):
            self.prev_x = pos_x
        delta_x = pos_x - self.prev_x
        reward = delta_x * 0.5  # strong incentive to move right

        # Upward progress bonus (jumping/climbing)
        if not hasattr(self, "prev_y"):
            self.prev_y = pos_y
        delta_y = pos_y - self.prev_y
        if delta_y > 0:
            reward += delta_y * 0.3  # reward climbing/jumping

        # Bonus for upward velocity (encourage jump)
        if vel_y > 1.0:
            reward += 1.0

        # Small step penalty to encourage speed
        reward -= 0.01

        # Death penalty
        done = False
        if dead:
            reward -= 20.0
            done = True

        # Clip reward to avoid spikes
        reward = np.clip(reward, -20, 20)

        # Update previous positions
        self.prev_x = pos_x
        self.prev_y = pos_y

        return reward, done





    def render(self, mode="human"):
        """Optional visualization (skip for now)."""
        pass


    def close(self):
        """Clean shutdown."""
        try:
            self.client.close()
        except:
            pass
        try:
            self.server.close()
        except:
            pass

if __name__ == "__main__":
    print("Creating CelesteEnv...")
    env = CelesteEnv()
    print("Environment created!")

    # Try a quick test step
    try:
        obs, info = env.reset()
        print("Initial observation received!")

        while True:
            obs, reward, done, _, _ = env.step(env.action_space.sample())
            print(f"Reward: {reward:.2f}, Done: {done}")
            if done:
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("🛑 Interrupted, closing connection.")
    finally:
        env.close()
        print("Connection closed.")
