import os
import socket
import pickle
import threading
import torch

import gr00t
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "/home/josyula/projects/gr00t_checkpoint/checkpoint-100000"  
EMBODIMENT_TAG = "new_embodiment"
DATA_CONFIG = "so100"   # matches your training dataset config

HOST = "0.0.0.0"
PORT = 5001  # GR00T server port

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load Policy once
# ----------------------------
print("[Server] Loading GR00T policy...")
data_config = DATA_CONFIG_MAP[DATA_CONFIG]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)
print("[Server] Policy loaded successfully.")

# ----------------------------
# Client Handler
# ----------------------------
def handle_client(conn, addr):
    print(f"[Server] Connection from {addr}")
    try:
        data = b""
        while True:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet
            if data.endswith(b"<END>"):
                data = data[:-5]
                break

        observation = pickle.loads(data)

        # Run inference
        with torch.no_grad():
            result = policy.get_action(observation)

        response = pickle.dumps(result)
        conn.sendall(response + b"<END>")

    except Exception as e:
        print(f"[Server] Error handling client {addr}: {e}")
    finally:
        conn.close()
        print(f"[Server] Connection closed: {addr}")

# ----------------------------
# Main Server Loop
# ----------------------------
def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"[Server] Listening on {HOST}:{PORT}")

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
        thread.start()

if __name__ == "__main__":
    main()
