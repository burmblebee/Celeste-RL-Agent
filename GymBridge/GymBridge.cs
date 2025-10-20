extern alias celeste;
extern alias everest;
using Microsoft.Xna.Framework;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using CelesteBase = celeste::Celeste;
using EverestAPI = everest::Celeste.Mod;

namespace GymBridge
{
    public class GymBridgeModule : EverestAPI.EverestModule
    {
        public static GymBridgeModule Instance;

        private TcpClient client;
        private NetworkStream stream;
        private Thread listenerThread;
        private int pendingAction = 0;

        // Progress tracking
        private float cumulativeReward = 0f;
        private int deathCount = 0;
        private readonly string progressPath = Path.Combine("GymBridgeProgress.json");

        public GymBridgeModule() => Instance = this;

        public override void Load()
        {
            EverestAPI.Logger.Log(EverestAPI.LogLevel.Info, "GymBridge", "Loaded ✅");
            On.Celeste.Player.Update += Player_Update;

            try
            {
                client = new TcpClient("127.0.0.1", 5000);
                stream = client.GetStream();
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Info, "GymBridge", "Connected to Python agent ✅");

                listenerThread = new Thread(ListenForActions);
                listenerThread.IsBackground = true;
                listenerThread.Start();
            }
            catch
            {
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge", "Python agent not running ⚠️");
                client = null;
                stream = null;
            }

            LoadProgress();
        }

        public override void Unload()
        {
            On.Celeste.Player.Update -= Player_Update;
            SaveProgress();

            try { stream?.Close(); } catch { }
            try { client?.Close(); } catch { }
            if (listenerThread != null && listenerThread.IsAlive)
                listenerThread.Abort();
        }

        private void Player_Update(On.Celeste.Player.orig_Update orig, CelesteBase.Player self)
        {
            float posX = self.Position.X;
            float posY = self.Position.Y;

            ApplyAction(self, pendingAction);
            orig(self);

            var level = self.Scene as CelesteBase.Level;
            if (level == null) return;

            posX = self.Position.X;
            posY = self.Position.Y;
            float velX = self.Speed.X;
            float velY = self.Speed.Y;
            bool canDash = self.CanDash;
            bool onGround = self.OnGround();
            bool onWall = self.CollideCheck<CelesteBase.Solid>(self.Position + new Vector2(-1, 0))
                       || self.CollideCheck<CelesteBase.Solid>(self.Position + new Vector2(1, 0));
            float stamina = self.Stamina;
            bool dead = self.Dead;

            if (dead)
            {
                deathCount++;
                cumulativeReward -= 10f;
                SaveProgress();
            }

            // Vision grid
            int gridWidth = 13;
            int gridHeight = 7;
            float spacingX = 104f; // ~1/3 screen horizontally
            float spacingY = 60f;  // ~1/3 vertically
            int halfW = gridWidth / 2;
            int halfH = gridHeight / 2;

            List<float> solidGrid = new List<float>();
            List<float> dangerGrid = new List<float>();

            foreach (var dy in Enumerable.Range(-halfH, gridHeight))
            {
                foreach (var dx in Enumerable.Range(-halfW, gridWidth))
                {
                    Vector2 checkPos = self.Position + new Vector2(dx * spacingX, dy * spacingY);

                    bool solid = self.CollideCheck<CelesteBase.Solid>(checkPos);
                    solidGrid.Add(solid ? 1f : 0f);

                    bool danger = false;
                    foreach (var entity in level.Entities)
                    {
                        if (entity is CelesteBase.Spikes spikes)
                        {
                            Rectangle rect = new Rectangle(
                                (int)spikes.Position.X,
                                (int)spikes.Position.Y,
                                (int)spikes.Width,
                                (int)spikes.Height
                            );
                            if (rect.Contains((int)checkPos.X, (int)checkPos.Y))
                            {
                                danger = true;
                                break;
                            }
                        }
                    }
                    dangerGrid.Add(danger ? 1f : 0f);
                }
            }

            List<float> features = new List<float>
            {
                posX, posY, velX, velY,
                canDash ? 1 : 0,
                onGround ? 1 : 0,
                onWall ? 1 : 0,
                dead ? 1 : 0,
                stamina
            };
            features.Add(grabToggled ? 1 : 0); // isGrabbing
            features.AddRange(solidGrid);
            features.AddRange(dangerGrid);

            if (stream != null && stream.CanWrite)
            {
                try
                {
                    string json = JsonConvert.SerializeObject(new { features });
                    byte[] data = Encoding.UTF8.GetBytes(json + "\n");
                    stream.Write(data, 0, data.Length);
                }
                catch (SocketException ex)
                {
                    EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge", $"Socket error: {ex.Message}");
                    try { stream?.Close(); } catch { }
                    try { client?.Close(); } catch { }
                    stream = null;
                    client = null;
                }
            }
        }

        private bool grabToggled = false;

        private void ApplyAction(CelesteBase.Player self, int action)
        {
            switch (action)
            {
                case 1:
                    self.MoveH(-1); // Move left
                    break;

                case 2:
                    self.MoveH(1); // Move right
                    break;

                case 3:
                    if (self.OnGround())
                        self.Jump();
                    break;

                case 4:
                    if (self.CanDash)
                    {
                        self.DashDir = Vector2.UnitX;
                        self.StartDash();
                    }
                    break;

                case 5:
                    // Toggle grab mode
                    grabToggled = !grabToggled;
                    break;
            }

            // Handle grab/climb toggle behavior
            if (grabToggled)
            {
                bool touchingWall = self.CollideCheck<CelesteBase.Solid>(self.Position + new Vector2(-1, 0)) ||
                                    self.CollideCheck<CelesteBase.Solid>(self.Position + new Vector2(1, 0));

                // Only enter climb if touching a wall and not already climbing
                if (touchingWall && self.StateMachine.State != CelesteBase.Player.StClimb)
                {
                    self.StateMachine.State = CelesteBase.Player.StClimb;
                }
                else if (!touchingWall && self.StateMachine.State == CelesteBase.Player.StClimb)
                {
                    self.StateMachine.State = CelesteBase.Player.StNormal;
                }
            }
            else
            {
                // Release grab if toggled off and currently climbing
                if (self.StateMachine.State == CelesteBase.Player.StClimb)
                {
                    self.StateMachine.State = CelesteBase.Player.StNormal;
                }
            }
        }


        private void ListenForActions()
        {
            byte[] buffer = new byte[1024];
            StringBuilder sb = new StringBuilder();

            try
            {
                while (true)
                {
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    if (bytesRead <= 0) break;

                    sb.Append(Encoding.UTF8.GetString(buffer, 0, bytesRead));
                    while (sb.ToString().Contains("\n"))
                    {
                        string line = sb.ToString().Split('\n')[0];
                        sb.Remove(0, sb.ToString().IndexOf("\n") + 1);

                        try
                        {
                            var msg = JsonConvert.DeserializeObject<Dictionary<string, int>>(line);
                            if (msg != null && msg.ContainsKey("action"))
                                pendingAction = msg["action"];
                        }
                        catch { }
                    }
                }
            }
            catch (Exception ex)
            {
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge", $"Listener error: {ex.Message}");
            }
        }

        private void SaveProgress()
        {
            var data = new
            {
                cumulativeReward,
                deathCount,
                timestamp = DateTime.Now
            };

            File.WriteAllText(progressPath, JsonConvert.SerializeObject(data, Formatting.Indented));
        }

        private void LoadProgress()
        {
            if (File.Exists(progressPath))
            {
                try
                {
                    var json = File.ReadAllText(progressPath);
                    var data = JsonConvert.DeserializeObject<Dictionary<string, object>>(json);
                    cumulativeReward = Convert.ToSingle(data["cumulativeReward"]);
                    deathCount = Convert.ToInt32(data["deathCount"]);
                }
                catch { }
            }
        }
    }
}
