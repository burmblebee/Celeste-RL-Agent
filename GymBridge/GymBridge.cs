extern alias celeste;
extern alias CelesteTAS; // 👈 reference to your CelesteTAS-EverestInterop.dll
extern alias everest;
using CelesteTAS::TAS;
using CelesteTAS::TAS.Input;
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
using TASCommands = CelesteTAS::TAS.Input.Commands;
using TASInput = CelesteTAS::TAS.Input;
using CelesteTAS::TAS.Input;


namespace GymBridge
{
    public class GymBridgeModule : EverestAPI.EverestModule
    {
        public static GymBridgeModule Instance;

        private TcpClient client;
        private NetworkStream stream;
        private Thread listenerThread;
        private int pendingAction = 0;

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
            ApplyAction(self, pendingAction);
            orig(self);

            var level = self.Scene as CelesteBase.Level;
            if (level == null) return;

            SendObservation(level, self);

        }

        private bool grabToggled = false;

        private void ApplyAction(CelesteBase.Player self, int action)
        {
            bool left = false, right = false, up = false, down = false, jump = false, dash = false;
            if (action == 1) left = true;
            else if (action == 2) right = true;
            else if (action == 3) jump = true;
            else if (action == 4) dash = true;
            else if (action == 5) grabToggled = !grabToggled;

            // Build a fake TAS input line like the ones in .tas files
            string tasLine = BuildTASLine(left, right, up, down, jump, dash, grabToggled);

            if (InputFrame.TryParse(tasLine, "GymBridge", 0, 0, null, out var frame))
            {
                InputHelper.FeedInputs(frame);
            }
        }

        private string BuildTASLine(bool left, bool right, bool up, bool down, bool jump, bool dash, bool grab)
        {
            // Construct a TAS input string such as "1,1,1,1,1,1,1"
            // The syntax matches CelesteTAS format, e.g. "L, J, D, G"
            List<string> parts = new List<string>();
            if (left) parts.Add("L");
            if (right) parts.Add("R");
            if (up) parts.Add("U");
            if (down) parts.Add("D");
            if (jump) parts.Add("J");
            if (dash) parts.Add("X");
            if (grab) parts.Add("G");

            // Default to 1 frame if nothing else
            parts.Add("1");

            return string.Join(", ", parts);
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
        private void SendMessage(object obj)
        {
            if (stream == null || !stream.CanWrite)
                return;

            try
            {
                string json = JsonConvert.SerializeObject(obj);
                byte[] bytes = Encoding.UTF8.GetBytes(json + "\n");
                stream.Write(bytes, 0, bytes.Length);
                stream.Flush();
            }
            catch (Exception ex)
            {
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Warn, "GymBridge", $"Send error: {ex.Message}");
            }
        }

        private void SendObservation(CelesteBase.Level level, CelesteBase.Player player)
        {
            if (stream == null || !stream.CanWrite || player == null)
                return;

            var pos = player.Position;
            const int gridWidth = 16;
            const int gridHeight = 9;
            const int tileSize = 8;
            int halfW = gridWidth / 2;
            int halfH = gridHeight / 2;

            // Compute grid pixel bounds
            float originX = pos.X - halfW * tileSize;
            float originY = pos.Y - halfH * tileSize;
            float endX = pos.X + halfW * tileSize;
            float endY = pos.Y + halfH * tileSize;

            // Build grid
            char[,] grid = new char[gridHeight, gridWidth];
            for (int y = 0; y < gridHeight; y++)
                for (int x = 0; x < gridWidth; x++)
                    grid[y, x] = '.'; // empty space

            // Static solids (walls, ground, etc.)
            var solids = level.Tracker.GetEntities<CelesteBase.Solid>();
            foreach (CelesteBase.Solid s in solids)
            {
                var rect = s.Collider?.Bounds ?? Rectangle.Empty;
                for (int gx = 0; gx < gridWidth; gx++)
                {
                    for (int gy = 0; gy < gridHeight; gy++)
                    {
                        float worldX = originX + gx * tileSize + tileSize / 2f;
                        float worldY = originY + gy * tileSize + tileSize / 2f;
                        if (rect.Contains((int)worldX, (int)worldY))
                            grid[gy, gx] = '#';
                    }
                }
            }

            // Mark player center
            grid[halfH, halfW] = '@';

            // Convert grid to string array
            var gridStrings = Enumerable.Range(0, gridHeight)
                .Select(y => new string(Enumerable.Range(0, gridWidth).Select(x => grid[y, x]).ToArray()))
                .ToArray();

            // Build JSON observation
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
                GridOrigin = new
                {
                    X = originX,
                    Y = originY,
                    Width = gridWidth * tileSize,
                    Height = gridHeight * tileSize
                },
                Grid = gridStrings
            };

            try
            {
                string json = JsonConvert.SerializeObject(obs);
                byte[] msg = Encoding.UTF8.GetBytes(json + "\n");
                stream.Write(msg, 0, msg.Length);
            }
            catch (Exception e)
            {
                EverestAPI.Logger.Log(EverestAPI.LogLevel.Info, "GymBridge", $"Failed to send observation: {e}");
            }
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
