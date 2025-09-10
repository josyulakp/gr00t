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

import os
import glob


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
import pyrealsense2 as rs

from concurrent.futures import ThreadPoolExecutor

# Thread pool for async network calls
net_pool = ThreadPoolExecutor(max_workers=2)

#####################################################################
# Robot and Camera Setup
#####################################################################
robot = franky.Robot("172.16.0.2")
robot.relative_dynamics_factor = 0.03
robot.set_collision_behavior([30.0]*7, [30.0]*6)

gripper = franky.Gripper("172.16.0.2")
max_gripper_width = 0.08  # ~80 mm
FPS = 30
chunk = 20
blend = 10
time_delay = 3
FPS = 30
lipo = ActionLiPo(chunk_size=chunk, blending_horizon=blend, len_time_delay=time_delay)

# # Cameras setup
# camera_configs = {
#     "left": OpenCVCameraConfig(camera_index=16, fps=FPS, width=640, height=480),
#     "right": OpenCVCameraConfig(camera_index=10, fps=FPS, width=640, height=480),
#     "wrist": OpenCVCameraConfig(camera_index=4, fps=FPS, width=640, height=480),
# }
# cameras = {}
# for name, cfg in camera_configs.items():
#     cameras[name] = OpenCVCamera(cfg)
#     try:
#         cameras[name].connect()
#         cameras[name].async_read()
#         print(f"Connected to camera {name} (index {cfg.camera_index})")
#     except Exception as e:
#         print(f"Failed to connect camera {name}: {e}")


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


def create_rs_camera(serial, width=640, height=480, fps=30):
    """Initialize and return a RealSense pipeline bound to a specific serial, with depth stream."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipeline.start(config)

    # Align depth to color stream
    align = rs.align(rs.stream.color)
    return pipeline, align

def get_depth_scale(pipeline):
    # Get the active device from the pipeline
    profile = pipeline.get_active_profile()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    return depth_scale



action_queue = queue.Queue(maxsize=1)

def detect_wrist_pixels(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold dark regions (black gripper jaws)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    wrist_points = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter by region size (gripper pieces are tall/thin)
        if h > 40 and w < 100:
            cx, cy = x + w//2, y + h//2
            wrist_points.append((cx, cy))
            if debug:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.circle(image, (cx, cy), 5, (0,0,255), -1)
    
    if debug:
        cv2.imwrite("wrist_detected.png", image)
    
    return wrist_points

def detect_bowl_and_check_wrist(depth_image, wrist_px_list, depth_scale=0.001, debug=True):
    """
    Detect bowl rim via circle fitting on depth image and check if wrist is inside.
    inside = True if any wrist pixel depth < rim mean depth.

    Args:
        depth_image: np.ndarray (H,W), raw depth values.
        wrist_px_list: list of (x,y) wrist pixel coordinates in image space.
        depth_scale: depth unit scaling (to meters).
        debug: if True, saves visualization.
    """
    h, w = depth_image.shape
    gray = cv2.convertScaleAbs(depth_image, alpha=0.03)

    # Detect circles (bowl rim)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=75,
        maxRadius=200
    )

    inside = False
    circle_params = None

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circle_params = max(circles, key=lambda c: c[2])  # largest circle
        cx, cy, r = circle_params

        # Extract rim band (pixels near circle edge)
        yy, xx = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        rim_mask = np.logical_and(dist_from_center > r-5, dist_from_center < r+5)

        rim_depths = depth_image[rim_mask].astype(np.float32) * depth_scale
        rim_depth = np.nanmean(rim_depths) if rim_depths.size > 0 else np.inf


        # Wrist depth(s)
        for (wx, wy) in wrist_px_list:
            if 0 <= wx < w and 0 <= wy < h:
                wrist_depth = depth_image[wy, wx] * depth_scale
                if wrist_depth > rim_depth - 0.01:  # margin = 1cm
                    inside = True
                    break

        if debug:
            disp = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            cv2.circle(disp, (cx, cy), r, (0,255,0), 2)
            for (wx, wy) in wrist_px_list:
                cv2.circle(disp, (wx, wy), 5, (0,0,255), -1)
            cv2.imwrite("/home/prnuc/Documents/josyulak/gr00t/scripts/depth_wrist.png", disp)
            print(f"Rim depth={rim_depth:.3f}m, Inside={inside}")

    else:
        print("No bowl rim detected")

    return inside

def fetch_actions_loop(args):
    """ Continuously fetch actions from server with depth capture (non-blocking). """
    serial_left = "142422250807" 
    serial_top = "025522060843"
    serial_wrist = "218622278163"

    rs_left, align_left = create_rs_camera(serial_left)
    rs_top, align_top = create_rs_camera(serial_top)
    rs_wrist, align_wrist = create_rs_camera(serial_wrist)

    video_path = get_next_video_filename(ext="avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (640, 480))
    print(f"[Video] Recording top camera to {video_path}")

    depth_scale = get_depth_scale(rs_wrist)

    while True:
        try:
            # --- Non-blocking frame grab ---
            left_frames = rs_left.poll_for_frames()
            top_frames  = rs_top.poll_for_frames()
            wrist_frames = rs_wrist.poll_for_frames()

            if not (left_frames and top_frames and wrist_frames):
                time.sleep(0.01)
                continue  # skip if any camera has no new frames

            # Align to color stream
            left_frames = align_left.process(left_frames)
            top_frames  = align_top.process(top_frames)
            wrist_frames = align_wrist.process(wrist_frames)

            # Extract color
            left_img  = np.asanyarray(left_frames.get_color_frame().get_data())
            top_img   = np.asanyarray(top_frames.get_color_frame().get_data())
            wrist_img = np.asanyarray(wrist_frames.get_color_frame().get_data())

            # Extract depth (aligned)
            wrist_depth = np.asanyarray(wrist_frames.get_depth_frame().get_data())

            # Quick sanity checks
            if wrist_depth is None or left_img is None or top_img is None or wrist_img is None:
                continue
            if ((np.mean(wrist_img) <= 50.0) or (np.mean(top_img) <= 50.0)):
                continue
            ##
            # Save debug images
            cv2.imwrite("left_image.png", left_img)
            cv2.imwrite("top_image.png", top_img)
            cv2.imwrite("wrist_image.png", wrist_img)
            cv2.imwrite("wrist_depth.png", (wrist_depth * 0.05).astype(np.uint8))


            video_writer.write(top_img)

            wrist_pixels  =  detect_wrist_pixels(wrist_img, debug=True)
            inside_bowl = detect_bowl_and_check_wrist(wrist_depth, wrist_pixels, depth_scale=depth_scale, debug=True)

            # Robot state
            robot_state   = robot.current_joint_state.position.reshape(1, 7)
            gripper_state = np.array(gripper.width / 2.0).reshape(1, 1)

            # Observation dict
            obs = {
                "video.left":  left_img.reshape(1, 480, 640, 3),
                "video.right": top_img.reshape(1, 480, 640, 3),
                "video.wrist": wrist_img.reshape(1, 480, 640, 3),
                "state.single_arm": robot_state,
                "state.gripper": gripper_state,
                "annotation.human.action.task_description": [
                    "Pick the bowl and place it in the green square"
                ],
            }

            # Depth heuristic (average of bottom quadrants)
            h, w = wrist_depth.shape[:2]
            half_h, half_w = h // 2, w // 2
            bottom_left  = wrist_depth[half_h:h, 0:half_w]
            bottom_right = wrist_depth[half_h:h, half_w:w]
            depth_mean = ((bottom_left.mean() + bottom_right.mean()) * 0.5) * depth_scale

            # Query policy
            if args.http_server:
                print("2")
                actions = _example_http_client_call(obs, args.host, args.port, args.api_token)
                print("3")
            else:
                actions = _example_zmq_client_call(obs, args.host, args.port, args.api_token)

            actions["depth_mean"] = depth_mean
            actions["inside_bowl"] = inside_bowl

            # Replace queue contents with latest actions (non-blocking put)
            if not action_queue.empty():
                try:
                    action_queue.get_nowait()
                except queue.Empty:
                    pass
            action_queue.put_nowait(actions)

        except Exception as e:
            print(f"[Fetcher] Error: {e}")
            time.sleep(0.01)


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

def get_next_video_filename(base_dir="videos", prefix="experiment", ext="mp4"):
    os.makedirs(base_dir, exist_ok=True)
    existing = glob.glob(os.path.join(base_dir, f"{prefix}[0-9][0-9][0-9].{ext}"))
    numbers = [int(os.path.basename(f).replace(prefix, "").replace(f".{ext}", "")) for f in existing]
    next_number = max(numbers) + 1 if numbers else 1
    return os.path.join(base_dir, f"{prefix}{next_number:03d}.{ext}")


def execute_actions_loop(fps=30):
    """Execute actions and allow interleaving using LiPo blending."""
    dt = 1.0 / fps
    current_chunk = None
    current_gripper = None
    playhead = 0

    # Track when we last closed the gripper
    last_closed_time = None
    hold_duration = 10.0  # seconds

    if gripper.width / 2 <= 0.03: 
        gripper.open(1.0)

    depth_mean = 10.0
    inside_bowl = False

    while True:
        try:
            # === Check if new chunk arrives ===
            try:
                new_chunk = action_queue.get_nowait()
                new_actions = np.array(new_chunk["action.single_arm"])
                new_gripper = np.array(new_chunk["action.gripper"])
                print("gripper actions", new_gripper)
                depth_mean = new_chunk.get("depth_mean", depth_mean)
                inside_bowl = new_chunk.get("inside_bowl", inside_bowl)
                print("execute depth mean ", depth_mean)

                # downsample + smooth
                new_actions = new_actions[::2]
                new_gripper = new_gripper[::2]
                new_actions = moving_average_chunk(new_actions, window_size=2)

                # Blend new chunk
                if current_chunk is not None:
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

            except queue.Empty:
                pass

            if current_chunk is None or current_gripper is None:
                time.sleep(dt)
                continue

            if playhead >= len(current_chunk):
                time.sleep(dt)
                continue

            # === Execute one step ===
            action = current_chunk[playhead]
            robot.move(franky.JointMotion(action.tolist()), asynchronous=True)
            # === Gripper logic ===
            now = time.time()

            # Trigger closing (if action suggests or depth threshold hit)int(current_gripper.shape[0]/2)
            if (depth_mean <=0.11 or current_gripper[-3:].mean() <= 0.03) and last_closed_time is None:
                gripper.grasp_async(0.0, 5.0, 30.0, epsilon_outer=0.01)
                last_closed_time = now
                print("[Gripper] CLOSE triggered")

            # Release only after 5 seconds have passed
            if last_closed_time is not None and (now - last_closed_time) >= hold_duration:
                # if current_gripper[int(current_gripper.shape[0]/2):].mean() > 0.02:
                    gripper.open_async(1.0)
                    print("[Gripper] OPEN triggered after hold")
                    last_closed_time = None  # reset

            playhead += 1
            time.sleep(dt)

        except Exception as e:
            print(f"[Executor] Error executing action: {e}")
            if robot.has_errors: 
                robot.recover_from_errors()
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

 