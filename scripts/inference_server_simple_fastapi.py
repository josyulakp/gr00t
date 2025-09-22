#!/usr/bin/env python3

import os
import io
import time
import base64
import pickle
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

import gr00t
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "/home/josyula/projects/gr00t_checkpoint/checkpoint-100000"
EMBODIMENT_TAG = "new_embodiment"
DATA_CONFIG = "so100"   # matches your training dataset config

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
# FastAPI App
# ----------------------------
app = FastAPI(title="GR00T Inference Server")

def encode_result(result_obj):
    """Serialize with pickle + base64 for safe transport."""
    pickled = pickle.dumps(result_obj)
    return base64.b64encode(pickled).decode("utf-8")

def decode_observation(obs_b64: str):
    """Decode base64 pickle observation from client."""
    data = base64.b64decode(obs_b64)
    return pickle.loads(data)

@app.post("/infer")
async def infer(observation_b64: str = Form(...)):
    """
    Run inference on an observation.
    The client should send a base64-encoded pickle of the observation.
    """
    try:
        start = time.time()
        observation = decode_observation(observation_b64)

        with torch.no_grad():
            result = policy.get_action(observation)

        response = {
            "success": True,
            "time_taken_sec": round(time.time() - start, 3),
            "result_b64": encode_result(result),  # send back as base64-pickle
        }
        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=500
        )
