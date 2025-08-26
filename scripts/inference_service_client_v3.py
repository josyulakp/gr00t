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
import pyrealsense2 as rs


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

def detect_bowl_and_check_wrist(depth_image, wrist_px=None, debug=False):
    """
    Fit circle to bowl rim in depth image and check if wrist is inside.
    wrist_px: (x,y) pixel of wrist in depth map. If None, use image center.
    """
    h, w = depth_image.shape
    gray = cv2.convertScaleAbs(depth_image, alpha=0.03)  # normalize
    # blur = cv2.GaussianBlur(gray, (9,9), 2)

    # Edge detection
    # edges = cv2.Canny(blur, 50, 150)

    # Detect circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=50,
        maxRadius=200
    )
    print("Circles: ", circles)
    inside = False
    circle_params = None

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Take the largest circle as bowl
        circle_params = max(circles, key=lambda c: c[2])  # (x,y,r)
        cx, cy, r = circle_params

        # Wrist point
        if wrist_px is None:
            wrist_px = (w//2, h//2)
        wx, wy = wrist_px

        dist = np.sqrt((wx - cx)**2 + (wy - cy)**2)
        inside = dist < r

        if debug:
            disp = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            cv2.circle(disp, (cx, cy), r, (0,255,0), 2)
            cv2.circle(disp, wrist_px, 5, (0,0,255), -1)
            cv2.imwrite("/home/prnuc/Documents/josyulak/gr00t/scripts/depth_wrist.png", disp)
            # print(f"Circle center=({cx},{cy}), radius={r}, wrist={wrist_px}, inside={inside}")

    return inside

def inside_bowl(depth_image, depth_scale, roi=None, debug=False, crop_ratio=0.9):
    """
    Heuristic: compare mean depth in the center region to the rim region.
    If the center is closer to camera than rim, assume wrist is inside bowl.

    Args:
        depth_image: np.ndarray, depth map (H, W).
        depth_scale: scale to convert depth to meters.
        roi: unused (kept for compatibility).
        debug: if True, saves visualization.
        crop_ratio: fraction of image size to use as center crop.
    """
    h, w = depth_image.shape
    
    print("mean depth ", depth_scale*depth_image.mean())
    # Define a central crop
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    x0, y0 = w // 2 - cw // 2, h // 2 - ch // 2
    center_region = depth_image[y0:y0+ch, x0:x0+cw].astype(np.float32) * depth_scale

    # Rim = full image minus center
    mask = np.ones_like(depth_image, dtype=np.uint8)
    mask[y0:y0+ch, x0:x0+cw] = 0
    rim_region = depth_image[mask.astype(bool)].astype(np.float32) * depth_scale

    center_mean = np.nanmean(center_region)
    rim_mean = np.nanmean(rim_region)

    inside = center_mean < rim_mean - 0.02  # margin (2cm)

    if debug:
        disp = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        cv2.rectangle(disp, (x0, y0), (x0+cw, y0+ch), (0,255,0), 2)
        # out_path = "depth_debug.png"
        out_path = "/home/prnuc/Documents/josyulak/gr00t/scripts/depth_wrist.png"
        cv2.imwrite(out_path, disp)
        # print(f"[Debug] Saved visualization to {out_path}")
        print(f"Center mean depth: {center_mean:.3f} m, Rim mean: {rim_mean:.3f} m, Inside: {inside}")

    return inside

def fetch_actions_loop(args):
    """ Continuously fetch actions from server with depth capture """
    serial_left = "142422250807" 
    serial_top = "025522060843"
    serial_wrist = "218622278163"

    rs_left, align_left = create_rs_camera(serial_left)
    rs_top, align_top = create_rs_camera(serial_top)
    rs_wrist, align_wrist = create_rs_camera(serial_wrist)

    while True:
        try:
            # Wait + align frames
            left_frames = align_left.process(rs_left.wait_for_frames())
            top_frames = align_top.process(rs_top.wait_for_frames())
            wrist_frames = align_wrist.process(rs_wrist.wait_for_frames())

            # Extract color
            left_img = np.asanyarray(left_frames.get_color_frame().get_data())
            top_img = np.asanyarray(top_frames.get_color_frame().get_data())
            wrist_img = np.asanyarray(wrist_frames.get_color_frame().get_data())
            # wrist_points = detect_wrist_pixels(wrist_img)
            # wrist_points = [(494, 354), (532, 346), (423, 218), (560, 195)]
            # print(wrist_points)

            # Extract depth (aligned to color)
            # left_depth = np.asanyarray(left_frames.get_depth_frame().get_data())
            # top_depth = np.asanyarray(top_frames.get_depth_frame().get_data())
            wrist_depth = np.asanyarray(wrist_frames.get_depth_frame().get_data())

            if((np.mean(wrist_img, axis=(0,1))==0.0).any() or (np.mean(left_img, axis=(0,1))==0.0).any() or (np.mean(top_img, axis=(0,1))==0.0).any() or (np.mean(top_img, axis=(0,1))<=50.0).any()  ):
                #no action
                raise Exception("invalid image data")

            # Quick depth sanity check
            if wrist_depth is None:
                raise Exception("invalid depth data")

            # Save debug frames
            cv2.imwrite("/home/prnuc/Documents/josyulak/gr00t/scripts/left_image.png", left_img)
            cv2.imwrite("/home/prnuc/Documents/josyulak/gr00t/scripts/top_image.png", top_img)
            cv2.imwrite("/home/prnuc/Documents/josyulak/gr00t/scripts/wrist_image.png", wrist_img)
            cv2.imwrite("/home/prnuc/Documents/josyulak/gr00t/scripts/wrist_depth.png", (wrist_depth * 0.05).astype(np.uint8))  # scaled for visibility
            
            # Robot state
            robot_state = robot.current_joint_state.position.reshape(1, 7)
            gripper_state = np.array(gripper.width / 2.0).reshape(1, 1)

            # Build obs (RGB + depth)
            obs = {
                "video.left": left_img.reshape(1, 480, 640, 3),
                "video.right": top_img.reshape(1, 480, 640, 3),
                "video.wrist": wrist_img.reshape(1, 480, 640, 3),
                "state.single_arm": robot_state,
                "state.gripper": gripper_state,
                "annotation.human.action.task_description": [
                    "Pick the bowl and place it in the green square"
                ],
            }
            depth_scale = get_depth_scale(rs_wrist)
            depth_mean = depth_scale*wrist_depth.mean()
            # inside = inside_bowl(wrist_depth, depth_scale, debug=True)

            # inside = detect_bowl_and_check_wrist(wrist_depth, wrist_px=wrist_points[0], debug=True)
            # print("Inside ", inside)
            # Send obs to server
            if args.http_server:
                actions = _example_http_client_call(obs, args.host, args.port, args.api_token)
            else:
                actions = _example_zmq_client_call(obs, args.host, args.port, args.api_token)
            
            actions['depth_mean'] = depth_mean

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
    playhead = 0

    # Track when we last closed the gripper
    last_closed_time = None
    hold_duration = 4.0  # seconds

    if gripper.width / 2 <= 0.03: 
        gripper.open(1.0)

    depth_mean = 10.0

    while True:
        try:
            # === Check if new chunk arrives ===
            try:
                new_chunk = action_queue.get_nowait()
                new_actions = np.array(new_chunk["action.single_arm"])
                new_gripper = np.array(new_chunk["action.gripper"])
                depth_mean = new_chunk.get("depth_mean", depth_mean)
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

            # Trigger closing (if action suggests or depth threshold hit)
            if ((depth_mean <= 0.095) or current_gripper[int(current_gripper.shape[0]/2):].mean() <= 0.02) and last_closed_time is None:
                gripper.grasp_async(0.0, 1.0, 20.0, epsilon_outer=1.0)
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

 