# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

import time
from dataclasses import dataclass
from typing import Literal
import threading
from queue import Queue

import numpy as np
import tyro
import franky
import cv2
from opencv import OpenCVCameraConfig, OpenCVCamera
from scipy.spatial.transform import Rotation as R

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from action_lipo import ActionLiPo
from real_time_chunking import realtime_action

### === CONFIGURATION === ###
chunk = 50
blend = 10
time_delay = 3
FPS = 30
lipo = ActionLiPo(chunk_size=chunk, blending_horizon=blend, len_time_delay=time_delay)
robot = franky.Robot("172.16.0.2")
robot.relative_dynamics_factor = 0.02
robot.set_collision_behavior([15.0]*7, [30.0]*6)
gripper = franky.Gripper("172.16.0.2")
max_gripper_width = 0.08

camera_configs = {
    "left": OpenCVCameraConfig(camera_index=16, fps=FPS, width=640, height=480),
    "right": OpenCVCameraConfig(camera_index=10, fps=FPS, width=640, height=480),
    "wrist": OpenCVCameraConfig(camera_index=4, fps=FPS, width=640, height=480),
}
cameras = {}
for name, cfg in camera_configs.items():
    cam = OpenCVCamera(cfg)
    try:
        cam.connect()
        cam.async_read()
        cameras[name] = cam
        print(f"Connected to camera {name} (index {cfg.camera_index})")
    except Exception as e:
        print(f"Failed to connect camera {name}: {e}")

# === THREAD SHARED STATE === #
current_chunk = None
chunk_lock = threading.Lock()
stop_signal = False


### === HELPERS === ###

def get_lipo_actions(actions, prev_chunk):
    smoothed_actions = np.zeros_like(actions[:, :7])
    count = np.zeros(actions.shape[0])
    for start in range(0, actions.shape[0] - chunk + 1, 1):
        action_chunk = actions[start:start+chunk, :7]
        solved, _ = lipo.solve(action_chunk, prev_chunk, len_past_actions=blend if prev_chunk is not None else 0)
        if solved is not None:
            for i in range(chunk-blend):
                idx = start + i
                if idx < smoothed_actions.shape[0]:
                    smoothed_actions[idx] += solved[i]
                    count[idx] += 1
            prev_chunk = solved.copy()
        else:
            for i in range(chunk-blend):
                idx = start + i
                if idx < smoothed_actions.shape[0]:
                    smoothed_actions[idx] += action_chunk[i]
                    count[idx] += 1
            prev_chunk = action_chunk.copy()
    for i in range(smoothed_actions.shape[0]):
        if count[i] > 0:
            smoothed_actions[i] /= count[i]
        else:
            smoothed_actions[i] = actions[i, :7]
    return smoothed_actions

def action_fetcher(args, robot_state, gripper_state, top_img, wrist_img, left_img):
    global current_chunk
    obs = {
        "video.left":  left_img[:, :, ::-1].reshape(1, 480, 640, 3),
        "video.right": top_img[:, :, ::-1].reshape(1, 480, 640, 3),
        "video.wrist": wrist_img[:, :, ::-1].reshape(1, 480, 640, 3),
        "state.single_arm": robot_state,
        "state.gripper": gripper_state,
        "annotation.human.action.task_description": ["Pick the bowl and place it in the green square"],
    }
    if args.http_server:
        action = _example_http_client_call(obs, args.host, args.port, args.api_token)
    else:
        action = _example_zmq_client_call(obs, args.host, args.port, args.api_token)
    smoothed = get_lipo_actions(np.array(action["action.single_arm"]), None)
    with chunk_lock:
        current_chunk = smoothed

def action_executor():
    global current_chunk, stop_signal
    while not stop_signal:
        if current_chunk is not None:
            with chunk_lock:
                chunk_to_run = current_chunk.copy()
                current_chunk = None
            for i in range(chunk_to_run.shape[0]):
                robot.move(franky.JointMotion(chunk_to_run[i].tolist()))
                time.sleep(1.0 / FPS)
                with chunk_lock:
                    if current_chunk is not None:
                        break

### === CLIENT AND SERVER === ###

@dataclass
class ArgsConfig:
    model_path: str = "nvidia/GR00T-N1.5-3B"
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist"
    port: int = 5555
    host: str = "localhost"
    server: bool = False
    client: bool = False
    denoising_steps: int = 4
    api_token: str = None
    http_server: bool = False

def _example_zmq_client_call(obs: dict, host: str, port: int, api_token: str):
    policy_client = RobotInferenceClient(host=host, port=port, api_token=api_token)
    return policy_client.get_action(obs)

def _example_http_client_call(obs: dict, host: str, port: int, api_token: str):
    import json_numpy
    json_numpy.patch()
    import requests
    response = requests.post(f"http://{host}:{port}/act", json={"observation": obs})
    if response.status_code == 200:
        return response.json()
    else:
        print(f"HTTP Error: {response.status_code} - {response.text}")
        return {}

def main(args: ArgsConfig):
    global stop_signal
    if args.server:
        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )
        if args.http_server:
            from gr00t.eval.http_server import HTTPInferenceServer
            server = HTTPInferenceServer(policy, port=args.port, host=args.host, api_token=args.api_token)
        else:
            server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token)
        server.run()

    elif args.client:
        gripper.open_async(1.0)
        time.sleep(2.0)
        print("Starting executor thread...")
        executor_thread = threading.Thread(target=action_executor)
        executor_thread.start()

        try:
            while True:
                try:
                    top_img = cameras["right"].async_read()
                    wrist_img = cameras["wrist"].async_read()
                    left_img = cameras["left"].async_read()

                    if ((np.mean(wrist_img, axis=(0, 1)) == 0.0).any() or
                        (np.mean(left_img, axis=(0, 1)) == 0.0).any() or
                        (np.mean(top_img, axis=(0, 1)) == 0.0).any() or
                        (np.mean(top_img, axis=(0, 1)) <= 50.0).any()):
                        continue

                    robot_state = robot.current_joint_state.position.reshape(1, 7)
                    gripper_state = np.array(gripper.width / 2.0).reshape(1, 1)

                    fetcher_thread = threading.Thread(
                        target=action_fetcher,
                        args=(args, robot_state, gripper_state, top_img, wrist_img, left_img)
                    )
                    fetcher_thread.start()
                    time.sleep(chunk / FPS)

                except Exception as e:
                    print(f"Error in main loop: {e}")

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            stop_signal = True
            executor_thread.join()

    else:
        raise ValueError("Please specify either --server or --client")

if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
