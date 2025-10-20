import socket
import json

HOST = '127.0.0.1'
PORT = 5000

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Listening on {HOST}:{PORT}...")

        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            buffer = ""

            while True:
                data = conn.recv(1024).decode("utf-8")
                if not data:
                    break

                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    try:
                        message = json.loads(line)
                        features = message.get("features")

                        if features:
                            # Debug print
                            print("Features:", features[:12], "...")
                            print("------")

                            # 🔥 Send features back to CelesteEnv
                            packet = {"features": features}
                            conn.sendall((json.dumps(packet) + "\n").encode("utf-8"))

                    except json.JSONDecodeError as e:
                        print("Failed to decode JSON:", e)

if __name__ == "__main__":
    start_server()
