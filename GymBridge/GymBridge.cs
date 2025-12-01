extern alias celeste;
extern alias everest;
extern alias CelesteTAS;

using celeste::Monocle;
using CelesteTAS::TAS;
using CelesteTAS::TAS.Input;
using Microsoft.Xna.Framework;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using CelesteBase = celeste::Celeste;
using EverestAPI = everest::Celeste.Mod;
using TASCommands = CelesteTAS::TAS.Input.Commands;
using TASController = CelesteTAS::TAS.Input.InputController;
using TASInput = CelesteTAS::TAS.Input;
using TASManager = CelesteTAS::TAS.Manager;

namespace GymBridge
{
    public class GymBridgeModule : EverestAPI.EverestModule
    {
        public static GymBridgeModule Instance;

        private TASController tasController = new TASController();
        private TcpClient client;
        private NetworkStream stream;
        private Thread listenerThread;
        private static FieldInfo grabToggleField;

        private float cumulativeReward = 0f;
        private int deathCount = 0;

        private static readonly string BaseSaveDir =
            @"C:\Users\abbyo\OneDrive\Documents\ai class\Final Project\GymBridge";

        private readonly string progressPath =
            Path.Combine(BaseSaveDir, "GymBridgeProgress.json");

        private readonly string demoPath =
            Path.Combine(BaseSaveDir, "GymBridgeDemo.json");

        private InputData latestInput = new InputData();

        private readonly List<RecordedFrame> demoFrames = new List<RecordedFrame>();
        private bool receivedExternalInput = false;
        private DateTime lastExternalInput = DateTime.MinValue;

        public GymBridgeModule() => Instance = this;

        public override void Load()
        {
            EverestAPI.Logger.Log(EverestAPI.LogLevel.Info, "GymBridge", "Loaded");
            On.Celeste.Player.Update += Player_Update;

            // Enable TAS system
            TASManager.EnableRun();
            EverestAPI.Logger.Log(EverestAPI.LogLevel.Info, "GymBridge", "TAS system enabled");

            // Reflection for grabToggle
            var inputType = typeof(celeste::Celeste.Input);
            grabToggleField = inputType.GetField("grabToggle", BindingFlags.NonPublic | BindingFlags.Static);

            if (grabToggleField == null)
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge", "Failed to find Celeste.Input.grabToggle");
            else
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Info, "GymBridge", "Found Celeste.Input.grabToggle");

            // Connect to Python agent
            try
            {
                client = new TcpClient("127.0.0.1", 5000);
                stream = client.GetStream();
                listenerThread = new Thread(ListenForActions) { IsBackground = true };
                listenerThread.Start();
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Info, "GymBridge", "Connected to Python agent");
            }
            catch
            {
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge", "Python agent not running");
            }

            try
            {
                Directory.CreateDirectory(BaseSaveDir);
                File.WriteAllText(progressPath, "{ \"test\": true }");
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Info, "GymBridge", "Write test executed");
            }
            catch (Exception e)
            {
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge",
                    $"Failed creating save directory: {e}");
            }

            LoadProgress();
        }

        public override void Unload()
        {
            On.Celeste.Player.Update -= Player_Update;

            try { stream?.Close(); } catch { }
            try { client?.Close(); } catch { }

            if (listenerThread != null && listenerThread.IsAlive)
                listenerThread.Abort();

            try
            {
                File.WriteAllText(demoPath,
                    JsonConvert.SerializeObject(demoFrames, Formatting.Indented));
            }
            catch (Exception e)
            {
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge", $"Failed to save demo: {e}");
            }
        }

        private int saveTickCounter = 0;

        private void Player_Update(On.Celeste.Player.orig_Update orig, CelesteBase.Player self)
        {
            var level = self.Scene as CelesteBase.Level;
            if (level == null) return;

            orig(self);
            ApplyDirectInputs(self, latestInput, level);
            SendObservation(level, self);

            if (++saveTickCounter >= 120)
            {
                saveTickCounter = 0;
                SaveProgress(self);
            }
        }

        private bool prevJump = false;
        private bool prevDash = false;
        private bool prevGrab = false;

        private void ApplyDirectInputs(celeste::Celeste.Player self, InputData input, celeste::Celeste.Level level)
        {
            const float jumpSpeed = -210f;

            self.MoveH(input.MoveX);

            bool wallLeft = self.CollideCheck<celeste::Celeste.Solid>(self.Position - Vector2.UnitX);
            bool wallRight = self.CollideCheck<celeste::Celeste.Solid>(self.Position + Vector2.UnitX);

            SetGrabToggle(true);
            if (input.MoveY != 0 && (wallLeft || wallRight))
            {
                self.MoveV(input.MoveY);
            }

            if (input.Jump && !prevJump)
            {
                if (self.OnGround())
                {
                    self.Jump();
                    self.Speed.Y = jumpSpeed;
                }
                else if (wallLeft || wallRight)
                {
                    float wallDir = wallRight ? 1f : -1f;
                    self.Speed.X = -wallDir * 120f;
                    self.Speed.Y = jumpSpeed;
                    self.Jump();
                    self.Stamina = Math.Max(0, self.Stamina - 20f);
                }
            }

            if (input.Dash && !prevDash && self.Dashes > 0 && !self.DashAttacking)
            {
                Vector2 dir = new Vector2(input.MoveX, input.MoveY);
                if (dir.LengthSquared() < 0.01f)
                    dir = self.Facing == celeste::Celeste.Facings.Right ? Vector2.UnitX : -Vector2.UnitX;

                dir.Normalize();
                self.DashDir = dir;
                self.StartDash();
                self.StateMachine.State = celeste::Celeste.Player.StDash;
                self.Speed = dir * 240f;
            }

            if (input.MoveX > 0.1f)
                self.Facing = celeste::Celeste.Facings.Right;
            else if (input.MoveX < -0.1f)
                self.Facing = celeste::Celeste.Facings.Left;

            prevJump = input.Jump;
            prevDash = self.DashAttacking ? true : input.Dash;
            prevGrab = input.Grab;
        }

        private static bool GetGrabToggle()
        {
            if (grabToggleField == null) return false;
            return (bool)grabToggleField.GetValue(null);
        }

        private static void SetGrabToggle(bool value)
        {
            if (grabToggleField != null)
                grabToggleField.SetValue(null, value);
        }

        private void ListenForActions()
        {
            byte[] buffer = new byte[1024];
            StringBuilder sb = new StringBuilder();

            try
            {
                while (true)
                {
                    int bytes = stream.Read(buffer, 0, buffer.Length);
                    if (bytes <= 0) break;

                    sb.Append(Encoding.UTF8.GetString(buffer, 0, bytes));

                    while (sb.ToString().Contains("\n"))
                    {
                        string line = sb.ToString().Split('\n')[0];
                        sb.Remove(0, sb.ToString().IndexOf("\n") + 1);

                        try
                        {
                            var msg = JsonConvert.DeserializeObject<Dictionary<string, float[]>>(line);
                            if (msg != null && msg.ContainsKey("actions"))
                            {
                                latestInput = new InputData(msg["actions"]);
                                receivedExternalInput = true;
                                lastExternalInput = DateTime.Now;
                            }
                        }
                        catch { }
                    }
                }
            }
            catch (Exception e)
            {
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge", $"Listener error: {e}");
            }
        }

        private bool UsingHumanInput() =>
            (DateTime.Now - lastExternalInput).TotalSeconds > 0.2;

        private void SaveProgress(CelesteBase.Player player)
        {
            try
            {
                var data = new
                {
                    player.Position.X,
                    player.Position.Y,
                    cumulativeReward,
                    deathCount,
                    timestamp = DateTime.Now,
                };

                string json = JsonConvert.SerializeObject(data, Formatting.Indented);

                string tmp = progressPath + ".tmp";
                File.WriteAllText(tmp, json);
                File.Copy(tmp, progressPath, true);
                File.Delete(tmp);

                EverestAPI.Logger.Log(EverestAPI.LogLevel.Info, "GymBridge",
                    $"Saved progress -> reward={cumulativeReward}, deaths={deathCount}");
            }
            catch (Exception e)
            {
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Error, "GymBridge", $"SAVE FAILED: {e}");
            }
        }

        private void LoadProgress()
        {
            if (!File.Exists(progressPath)) return;

            try
            {
                var json = File.ReadAllText(progressPath);
                var data = JsonConvert.DeserializeObject<Dictionary<string, object>>(json);
                cumulativeReward = Convert.ToSingle(data["cumulativeReward"]);
                deathCount = Convert.ToInt32(data["deathCount"]);
            }
            catch { }
        }

        private void SendObservation(CelesteBase.Level level, CelesteBase.Player player)
        {
            if (stream == null || !stream.CanWrite || player == null)
                return;

            bool exited = level.Transitioning || level.Completed;
            bool dead = player.Dead;

            var pos = player.Position;
            const int W = 16, H = 16, tile = 8;
            int halfW = W / 2, halfH = H / 2;

            float originX = pos.X - (W * tile) / 2f;
            float originY = pos.Y - (H * tile) / 2f;

            int[,] grid = new int[H, W];

            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    float wx = originX + x * tile;
                    float wy = originY + y * tile;
                    var rect = new Rectangle((int)wx - 2, (int)wy - 2, 4, 4);

                    if (level.CollideCheck<celeste::Celeste.Solid>(rect)) grid[y, x] = 1;
                    else if (level.CollideCheck<celeste::Celeste.Spikes>(rect)) grid[y, x] = 2;
                    else grid[y, x] = 0;
                }

            grid[halfH, halfW] = 9;

            var exitBlock = level.Tracker.GetEntities<celeste::Celeste.ExitBlock>()?.FirstOrDefault();
            Vector2? exitPos = exitBlock?.Position;

            var gridStrings = Enumerable.Range(0, H)
                .Select(y => string.Join("", Enumerable.Range(0, W).Select(x => grid[y, x])))
                .ToArray();

            var obs = new
            {
                Room = level.Session?.Level,
                Player = new
                {
                    X = pos.X,
                    Y = pos.Y,
                    Speed = new { player.Speed.X, player.Speed.Y },
                    Dashes = player.Dashes,
                    GrabToggled = player.Holding != null,
                    OnGround = player.OnGround(),
                    Facing = player.Facing.ToString()
                },
                Exit = exitPos.HasValue ? new { X = exitPos.Value.X, Y = exitPos.Value.Y } : null,
                Grid = gridStrings,
                Done = exited || dead,
                Reason = dead ? "death" : (exited ? "exit" : "none")
            };

            if (UsingHumanInput())
                demoFrames.Add(new RecordedFrame { Observation = obs, Input = latestInput });

            try
            {
                string json = JsonConvert.SerializeObject(obs);
                byte[] msg = Encoding.UTF8.GetBytes(json + "\n");
                stream.Write(msg, 0, msg.Length);
            }
            catch (Exception e)
            {
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge", $"Observation send fail: {e}");
            }
        }
    }

    public struct InputData
    {
        public float MoveX, MoveY;
        public bool Jump, Dash, Grab;

        public InputData(float[] arr)
        {
            MoveX = arr.Length > 0 ? arr[0] : 0f;
            MoveY = arr.Length > 1 ? arr[1] : 0f;
            Jump = arr.Length > 2 && arr[2] > 0.5f;
            Dash = arr.Length > 3 && arr[3] > 0.5f;
            Grab = arr.Length > 4 && arr[4] > 0.5f;
        }
    }

    public class RecordedFrame
    {
        public object Observation;
        public InputData Input;
    }
}
