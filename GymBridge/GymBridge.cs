extern alias celeste;
extern alias CelesteTAS;
extern alias everest;

using celeste::Monocle;
using CelesteTAS::TAS;
using CelesteTAS::TAS.Input;
using Microsoft.Xna.Framework;
using MonoMod.Utils;
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
        private readonly string progressPath = Path.Combine("GymBridgeProgress.json");

        private InputData latestInput = new InputData();

        public GymBridgeModule() => Instance = this;

        public override void Load()
        {
            EverestAPI.Logger.Log(EverestAPI.LogLevel.Info, "GymBridge", "Loaded ✅");
            On.Celeste.Player.Update += Player_Update;


            TASManager.EnableRun();
            EverestAPI.Logger.Log(EverestAPI.LogLevel.Info, "GymBridge", "TAS system enabled ✅");

            // Access the private static field 'grabToggle' in Celeste.Input
            var inputType = typeof(celeste::Celeste.Input);
            grabToggleField = inputType.GetField("grabToggle", BindingFlags.NonPublic | BindingFlags.Static);

            if (grabToggleField == null)
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge", "Failed to find Celeste.Input.grabToggle ⚠️");
            else
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Info, "GymBridge", "Found Celeste.Input.grabToggle ✅");


            try
            {
                client = new TcpClient("127.0.0.1", 5000);
                stream = client.GetStream();
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Info, "GymBridge", "Connected to Python agent ✅");

                listenerThread = new Thread(ListenForActions)
                {
                    IsBackground = true
                };
                listenerThread.Start();
            }
            catch
            {
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge", "Python agent not running ⚠️");
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
            var level = self.Scene as CelesteBase.Level;
            if (level == null)
                return;

            orig(self);
            ApplyDirectInputs(self, latestInput, level);


            SendObservation(level, self);
        }

        private bool prevJump = false;
        private bool prevDash = false;
        private bool prevGrab = false;

        private void ApplyDirectInputs(celeste::Celeste.Player self, InputData input, celeste::Celeste.Level level)
        {
            const float runAccel = 1000f;
            const float maxRunSpeed = 90f;
            const float airAccel = 400f;
            const float jumpSpeed = -210f;
            const float gravity = 900f;

            // --- Movement ---
            //float targetSpeedX = input.MoveX * maxRunSpeed;
            //float accel = self.OnGround() ? runAccel : airAccel;
            //self.Speed.X = Calc.Approach(self.Speed.X, targetSpeedX, accel * (float)Engine.DeltaTime);
            self.MoveH(input.MoveX);


            // Apply gravity
            //if (!self.OnGround() && !input.Grab)
            //    self.Speed.Y = Calc.Approach(self.Speed.Y, 160f, gravity * (float)Engine.DeltaTime);

            bool wallOnLeft = self.CollideCheck<celeste::Celeste.Solid>(self.Position - Vector2.UnitX);
            bool wallOnRight = self.CollideCheck<celeste::Celeste.Solid>(self.Position + Vector2.UnitX);
            bool touchingWall = wallOnLeft || wallOnRight;

            // --- Climb ---
            //SetGrabToggle(input.Grab);
            SetGrabToggle(true);
            if(input.MoveY != 0 && touchingWall)
            {
                //self.Speed.Y = input.MoveY * 60f;
                self.MoveV(input.MoveY);
            }


            // --- Jump ---
            if (input.Jump && !prevJump)
            {
                if (self.OnGround())
                {
                    self.Jump();
                    self.Speed.Y = jumpSpeed;
                }
                else
                {
                    bool wallLeft = self.CollideCheck<celeste::Celeste.Solid>(self.Position - Vector2.UnitX);
                    bool wallRight = self.CollideCheck<celeste::Celeste.Solid>(self.Position + Vector2.UnitX);

                    if (input.Grab && (wallLeft || wallRight))
                    {
                        float wallDir = wallRight ? 1f : -1f;
                        self.Speed.X = -wallDir * 120f;
                        self.Speed.Y = jumpSpeed;
                        self.Jump();
                        self.Stamina = Math.Max(0, self.Stamina - 20f);
                    }
                }
            }

            // --- Dash ---
            if (input.Dash && !prevDash && self.Dashes > 0 && !self.DashAttacking)
            {
                Vector2 dashDir = new Vector2(input.MoveX, input.MoveY);

                // If too small, use facing
                if (dashDir.LengthSquared() < 0.01f)
                    dashDir = self.Facing == celeste::Celeste.Facings.Right ? Vector2.UnitX : -Vector2.UnitX;

                if (Math.Abs(dashDir.X) < 0.05f) dashDir.X = 0f;
                if (Math.Abs(dashDir.Y) < 0.05f) dashDir.Y = 0f;

                dashDir.Normalize();
                self.DashDir = dashDir;
                self.StartDash();
                self.StateMachine.State = celeste::Celeste.Player.StDash;
                self.Speed = dashDir * 240f;
            }

            // --- Facing direction ---
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
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    if (bytesRead <= 0) break;

                    sb.Append(Encoding.UTF8.GetString(buffer, 0, bytesRead));

                    while (sb.ToString().Contains("\n"))
                    {
                        string line = sb.ToString().Split('\n')[0];
                        sb.Remove(0, sb.ToString().IndexOf("\n") + 1);

                        try
                        {
                            var msg = JsonConvert.DeserializeObject<Dictionary<string, float[]>>(line);
                            if (msg != null && msg.ContainsKey("actions"))
                                latestInput = new InputData(msg["actions"]);
                        }
                        catch (Exception ex)
                        {
                            EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge", $"Parse error: {ex}");
                        }
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
            if (!File.Exists(progressPath))
                return;

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

            bool exited = false;
            bool dead = player.Dead;

            if (level.Transitioning || level.Completed)
                exited = true;

            var pos = player.Position;
            const int gridWidth = 32, gridHeight = 32, tileSize = 8;
            int halfW = gridWidth / 2, halfH = gridHeight / 2;

            float originX = pos.X - halfW * tileSize;
            float originY = pos.Y - halfH * tileSize;

            var solids = level.Tracker.GetEntities<celeste::Celeste.Solid>();
            var spikes = level.Tracker.GetEntities<celeste::Celeste.Spikes>();

            int[,] grid = new int[gridHeight, gridWidth];

            // --- Terrain sampling ---
            for (int gy = 0; gy < gridHeight; gy++)
            {
                for (int gx = 0; gx < gridWidth; gx++)
                {
                    float worldX = originX + gx * tileSize + tileSize / 2f;
                    float worldY = originY + gy * tileSize + tileSize / 2f;
                    var checkPos = new Vector2(worldX, worldY);

                    if (level.CollideCheck<celeste::Celeste.Solid>(checkPos))
                        grid[gy, gx] = 1;
                    else if (level.CollideCheck<celeste::Celeste.Spikes>(checkPos))
                        grid[gy, gx] = 2;
                    else
                        grid[gy, gx] = 0;
                }
            }


            grid[halfH, halfW] = 9; // player marker (optional visual)

            // --- Compute approximate exit coordinate ---
            Rectangle bounds = level.Bounds;

            // Try to find a real ExitBlock entity first
            // --- Compute real exit coordinate if present ---
            var exitBlock = level.Tracker.GetEntities<celeste::Celeste.ExitBlock>()?.FirstOrDefault();
            Vector2? exitPos = null;

            if (exitBlock != null)
            {
                // Found a real exit block in the level
                exitPos = exitBlock.Position;
            }

            // --- Serialize ---
            var gridStrings = Enumerable.Range(0, gridHeight)
                .Select(y => string.Join("", Enumerable.Range(0, gridWidth).Select(x => grid[y, x].ToString())))
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

                Exit = exitPos.HasValue
                    ? new { X = exitPos.Value.X, Y = exitPos.Value.Y }
                    : null,

                Grid = gridStrings,
                Done = exited || dead,
                Reason = dead ? "death" : (exited ? "exit" : "none")
            };

            try
            {
                string json = JsonConvert.SerializeObject(obs);
                byte[] msg = Encoding.UTF8.GetBytes(json + "\n");
                stream.Write(msg, 0, msg.Length);
            }
            catch (Exception e)
            {
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge", $"Failed to send observation: {e}");
            }

        }

    }


    public struct InputData
    {
        public float MoveX;
        public float MoveY;
        public bool Jump;
        public bool Dash;
        public bool Grab;

        public InputData(float[] arr)
        {
            MoveX = arr.Length > 0 ? arr[0] : 0f;
            MoveY = arr.Length > 1 ? arr[1] : 0f;
            Jump = arr.Length > 2 && arr[2] > 0.5f;
            Dash = arr.Length > 3 && arr[3] > 0.5f;
            Grab = arr.Length > 4 && arr[4] > 0.5f;
        }
    }
}

