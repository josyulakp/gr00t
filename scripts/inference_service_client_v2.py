# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GR00T Inference Service - Modified Client
Fetches action chunks from server, executes on Franka with blending between chunks.
"""

import time
import threading
import queue
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro
import cv2
import franky
from opencv import OpenCVCameraConfig, OpenCVCamera

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

from action_lipo import ActionLiPo



#####################################################################
# Robot and Camera Setup
#####################################################################
robot = franky.Robot("172.16.0.2")
robot.relative_dynamics_factor = 0.02
robot.set_collision_behavior([15.0]*7, [30.0]*6)

gripper = franky.Gripper("172.16.0.2")
max_gripper_width = 0.08  # ~80 mm
FPS = 30
chunk = 20
blend = 10
time_delay = 3
FPS = 30
lipo = ActionLiPo(chunk_size=chunk, blending_horizon=blend, len_time_delay=time_delay)

# Cameras setup
camera_configs = {
    "left": OpenCVCameraConfig(camera_index=16, fps=FPS, width=640, height=480),
    "right": OpenCVCameraConfig(camera_index=10, fps=FPS, width=640, height=480),
    "wrist": OpenCVCameraConfig(camera_index=4, fps=FPS, width=640, height=480),
}
cameras = {}
for name, cfg in camera_configs.items():
    cameras[name] = OpenCVCamera(cfg)
    try:
        cameras[name].connect()
        cameras[name].async_read()
        print(f"Connected to camera {name} (index {cfg.camera_index})")
    except Exception as e:
        print(f"Failed to connect camera {name}: {e}")


#####################################################################
# Args
#####################################################################
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


#####################################################################
# Networking Helpers
#####################################################################
def _example_zmq_client_call(obs: dict, host: str, port: int, api_token: str):
    policy_client = RobotInferenceClient(host=host, port=port, api_token=api_token)
    
    time_start = time.time()
    action = policy_client.get_action(obs)
    time_taken = time.time() - time_start
    # print(f"[ZMQ] Inference time: {time_taken:.4f} seconds")

    return action

def _example_http_client_call(obs: dict, host: str, port: int, api_token: str):
    import json_numpy
    json_numpy.patch()
    import requests
    response = requests.post(f"http://{host}:{port}/act", json={"observation": obs})
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}


#####################################################################
# Producer–Consumer with Blending
#####################################################################
action_queue = queue.Queue(maxsize=1)

def fetch_actions_loop(args):
    """ Continuously fetch actions from server """
    while True:
        try:
            top_img = cameras["right"].async_read()
            wrist_img = cameras["wrist"].async_read()
            left_img = cameras["left"].async_read()
            robot_state = robot.current_joint_state.position.reshape(1, 7)
            gripper_state = np.array(gripper.width / 2.0).reshape(1, 1)
            if((np.mean(wrist_img, axis=(0,1))==0.0).any() or (np.mean(left_img, axis=(0,1))==0.0).any() or (np.mean(top_img, axis=(0,1))==0.0).any() or (np.mean(top_img, axis=(0,1))<=50.0).any()  ):
                #no action
                raise Exception("invalid image data")

            cv2.imwrite("/home/prnuc/Documents/josyulak/gr00t/scripts/top_image.png", top_img[:, :, ::-1])
            cv2.imwrite("/home/prnuc/Documents/josyulak/gr00t/scripts/wrist_image.png", wrist_img[:, :, ::-1])
            cv2.imwrite("/home/prnuc/Documents/josyulak/gr00t/scripts/left_image.png", left_img[:, :, ::-1])

            obs = {
                "video.left": left_img[:, :, ::-1].reshape(1, 480, 640, 3),
                "video.right": top_img[:, :, ::-1].reshape(1, 480, 640, 3),
                "video.wrist": wrist_img[:, :, ::-1].reshape(1, 480, 640, 3),
                "state.single_arm": robot_state,
                "state.gripper": gripper_state,
                "annotation.human.action.task_description": ["Pick the bowl and place it in the green square"],
            }

            if args.http_server:
                actions = _example_http_client_call(obs, args.host, args.port, args.api_token)
            else:
                actions = _example_zmq_client_call(obs, args.host, args.port, args.api_token)

            if not action_queue.empty():
                try:
                    action_queue.get_nowait()
                except queue.Empty:
                    pass
            action_queue.put_nowait(actions)

        except Exception as e:
            print(f"[Fetcher] Error getting actions: {e}")
            time.sleep(0.1)

def blend_chunks(old_chunk, new_chunk, blend_len=4):
    """ Linearly interpolate between the end of old_chunk and start of new_chunk """
    if old_chunk is None:
        return new_chunk
    blended = []
    for i in range(blend_len):
        alpha = (i + 1) / (blend_len + 1)
        blended.append((1 - alpha) * old_chunk[-1] + alpha * new_chunk[0])
    return np.vstack([old_chunk, np.array(blended), new_chunk])


def get_lipo_actions(actions, prev_chunk):
    
    # LiPo smoothing for all actions (overlapping chunks)
    smoothed_actions = np.zeros_like(actions[:, :7])
    count = np.zeros(actions.shape[0])
    for start in range(0, actions.shape[0] - chunk + 1, 1):  # stride 1 for full smoothing
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
            print(f"LiPo failed to solve chunk starting at {start}, using raw actions.")
            for i in range(chunk-blend):
                idx = start + i
                if idx < smoothed_actions.shape[0]:
                    smoothed_actions[idx] += action_chunk[i]
                    count[idx] += 1
            prev_chunk = action_chunk.copy()
    # Average overlapping results
    for i in range(smoothed_actions.shape[0]):
        if count[i] > 0:
            smoothed_actions[i] /= count[i]
        else:
            smoothed_actions[i] = actions[i, :7]
    print("Smoothed actions shape:", smoothed_actions.shape)
    return smoothed_actions


def moving_average_chunk(chunk, window_size=3, axis=0, mode='valid'):
    """
    Smooths the chunk using a moving average filter.

    Args:
        chunk (np.ndarray): The action chunk to smooth (e.g., shape [N, D]).
        window_size (int): The size of the moving average window.
        axis (int): The axis along which to apply the moving average.
        mode (str): 'valid' (default) or 'same' for output shape.

    Returns:
        np.ndarray: Smoothed chunk.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode=mode), axis, chunk)

def execute_actions_loop(fps=30):
    """Execute actions and allow interleaving using LiPo blending."""
    dt = 1.0 / fps
    current_chunk = None
    current_gripper = None
    # prev_chunk = None
    playhead = 0
    last_closed = 0
    
    if gripper.width/2<=0.03: 
        gripper.open(1.0)


    while True:
        try:
            # Check if new chunk arrives
            try:
                new_chunk = action_queue.get_nowait()
                new_actions = np.array(new_chunk["action.single_arm"])
                new_gripper = np.array(new_chunk["action.gripper"])

                # downsample + smooth
                new_actions = new_actions[::2]
                new_gripper = new_gripper[::2]
                new_actions = moving_average_chunk(new_actions, window_size=2)

                # If we were already executing, interleave with LiPo
                if current_chunk is not None:
                    # keep last executed pose
                    tail = current_chunk[max(playhead-1, 0):playhead+1]
                    lipo_blended = get_lipo_actions(new_actions, tail)
                    current_chunk = lipo_blended
                    current_gripper = new_gripper
                    playhead = 0
                    print("[Executor] Interleaved new chunk with LiPo blending")
                    
                else:
                    current_chunk = new_actions
                    current_gripper = new_gripper
                    playhead = 0
                    print("[Executor] Starting first chunk")
                # prev_chunk = current_chunk.copy()

            except queue.Empty:
                pass

            # Nothing to execute yet
            if current_chunk is None or current_gripper is None:
                time.sleep(dt)
                continue

            # End of current chunk
            if playhead >= len(current_chunk):
                time.sleep(dt)
                continue

            # Execute one step
            action = current_chunk[playhead]
            grip = current_gripper[playhead]
            print("action ", action)
            print("gripper", current_gripper)
            print("gripper ", current_gripper[int(current_gripper.shape[0]/2):].mean())
            robot.move(franky.JointMotion(action.tolist()), asynchronous=True)
            print("LAST CLOSED", last_closed)
            print("Playahead", playhead)
            # Trigger closing once if actions[5] > 2.4
            # if playhead == 5 and action[5] > 2.5:
            #     gripper.grasp_async(0.0, 1.0, 20.0, epsilon_outer=1.0)
            #     print("Triggered closing at action[5] > 2.4")

            if current_gripper[int(current_gripper.shape[0]/2):].mean() <= 0.02:
                gripper.grasp_async(0.0, 1.0, 20.0, epsilon_outer=1.0)
                last_closed += 1
                print("CLOSING 1")

            # if last_closed > 0:
            #     last_closed += 1
            
            if last_closed >= 10: 
                if current_gripper[int(current_gripper.shape[0]/2):].mean() > 0.02:
                    gripper.open_async(1.0)
                    last_closed = 0
                else:
                    gripper.grasp_async(0.0, 1.0, 20.0, epsilon_outer=1.0)
                    last_closed += 1
                    print("CLOSING 2")
            

            

            playhead += 1
            time.sleep(dt)

        except Exception as e:
            print(f"[Executor] Error executing action: {e}")
            time.sleep(0.1)

#####################################################################
# Entry Point
#####################################################################
def main(args: ArgsConfig):
    if args.client:
        fetch_thread = threading.Thread(target=fetch_actions_loop, args=(args,), daemon=True)
        exec_thread = threading.Thread(target=execute_actions_loop, daemon=True)
        fetch_thread.start()
        exec_thread.start()
        fetch_thread.join()
        exec_thread.join()
    else:
        raise ValueError("This modified version is for client mode only.")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)

 