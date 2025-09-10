#!/usr/bin/env python3
"""
GR00T Inference Service - Polymetis Client
Fetches action chunks from custom GR00T inference server,
blends between chunks (ActionLiPo), and streams joint targets
to Polymetis using a non-blocking joint impedance controller.
"""

import os
import sys
sys.path.insert(0, "/home/prnuc/polymetis/polymetis/python")  # ensure torchcontrol & polymetis are importable early
import cv2
import time
import threading
import queue
from dataclasses import dataclass
import socket
import pickle
import numpy as np
import torch

# Cameras (optional; unchanged from your version)
import pyrealsense2 as rs

# Polymetis
from polymetis import RobotInterface

# LiPo blending (unchanged)
from action_lipo import ActionLiPo

from polymetis import GripperInterface


#####################################################################
# Config
#####################################################################
FPS = 15                     # streaming rate to the controller
CHUNK = 20                   # action window size from server
BLEND = 10                   # blending horizon for LiPo
TIME_DELAY = 3               # LiPo time delay parameter

# LiPo setup
lipo = ActionLiPo(chunk_size=CHUNK, blending_horizon=BLEND, len_time_delay=TIME_DELAY)

@dataclass
class ArgsConfig:
    host: str = "localhost"     # GR00T inference server IP
    port: int = 5000            # GR00T inference server port
    client: bool = True
    poly_ip: str = "localhost"  # Polymetis controller server IP
    poly_port: int = 50051      # Polymetis controller server port
    enforce_version: bool = False   # set True if versions are guaranteed to match
    go_home_on_start: bool = True   # optionally go home at start
    hz: int = FPS                   # desired client loop rate (falls back to robot.hz if available)


#####################################################################
# Networking (Socket Client)
#####################################################################
def gr00t_socket_client_call(obs: dict, host: str, port: int):
    """Send observation to GR00T inference server via raw socket and receive actions."""
    data = pickle.dumps(obs) + b"<END>"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(data)

    # Receive response
    response = b""
    while True:
        packet = s.recv(4096)
        if not packet:
            break
        response += packet
        if response.endswith(b"<END>"):
            response = response[:-5]
            break
    s.close()

    actions = pickle.loads(response)
    return actions


#####################################################################
# Camera Setup (RealSense) – unchanged
#####################################################################
def create_rs_camera(serial, width=640, height=480, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align

def get_depth_scale(pipeline):
    profile = pipeline.get_active_profile()
    depth_sensor = profile.get_device().first_depth_sensor()
    return depth_sensor.get_depth_scale()


#####################################################################
# Queues
#####################################################################
action_queue = queue.Queue(maxsize=1)


#####################################################################
# Fetch Loop (video + state -> GR00T -> actions)
#####################################################################
def fetch_actions_loop(args: ArgsConfig, robot: RobotInterface, gripper:GripperInterface ):
    """Continuously fetch actions from server with depth capture."""
    # Update these to your actual serials
    serial_left = "142422250807"
    serial_top = "025522060843"
    serial_wrist = "218622278163"

    rs_left, align_left = create_rs_camera(serial_left, fps=FPS)
    rs_top, align_top = create_rs_camera(serial_top, fps=FPS)
    rs_wrist, align_wrist = create_rs_camera(serial_wrist, fps=FPS)

    depth_scale = get_depth_scale(rs_wrist)

    H, W = 480, 640  # as configured above

    while True:
        try:
            # Grab frames
            left_frames = rs_left.wait_for_frames()
            top_frames = rs_top.wait_for_frames()
            wrist_frames = rs_wrist.wait_for_frames()

            # Align to color
            left_frames = align_left.process(left_frames)
            top_frames = align_top.process(top_frames)
            wrist_frames = align_wrist.process(wrist_frames)

            left_color = left_frames.get_color_frame()
            top_color = top_frames.get_color_frame()
            wrist_color = wrist_frames.get_color_frame()
            wrist_depth = wrist_frames.get_depth_frame()

            if not (left_color and top_color and wrist_color and wrist_depth):
                continue

            left_img = np.asanyarray(left_color.get_data())
            top_img = np.asanyarray(top_color.get_data())
            wrist_img = np.asanyarray(wrist_color.get_data())
            wrist_depth_np = np.asanyarray(wrist_depth.get_data())

            if left_img is None or top_img is None or wrist_img is None:
                continue

            # Robot state from Polymetis
            state = robot.get_robot_state()
            q = np.array(state.joint_positions, dtype=np.float32).reshape(1, 7)

            # NOTE: Polymetis doesn't control the Franka gripper directly; keep your external
            # gripper bridge if needed. Here we pass a placeholder scalar.
            gripper_state = gripper.get_state()
            gripper_width_half = np.array([gripper_state.width], dtype=np.float32).reshape(1, 1)

            # Observation for GR00T
            obs = {
                "video.left":  left_img.reshape(1, H, W, 3),
                "video.right": top_img.reshape(1, H, W, 3),
                "video.wrist": wrist_img.reshape(1, H, W, 3),
                "state.single_arm": q,
                "state.gripper": gripper_width_half,
                "annotation.human.action.task_description": [
                    "Pick the bowl and place it in the green square"
                ],
            }

            cv2.imwrite("left_image.png", left_img)
            cv2.imwrite("top_image.png", top_img)
            cv2.imwrite("wrist_image.png", wrist_img)
            # cv2.imwrite("wrist_depth.png", (wrist_depth * 0.05).astype(np.uint8))


            # Get actions from server
            actions = gr00t_socket_client_call(obs, args.host, args.port)

            # Depth heuristic
            depth_mean = float(wrist_depth_np.mean() * depth_scale)
            actions["depth_mean"] = depth_mean

            # Replace queue contents (non-blocking)
            if not action_queue.empty():
                try:
                    action_queue.get_nowait()
                except queue.Empty:
                    pass
            action_queue.put_nowait(actions)

        except Exception as e:
            print(f"[Fetcher] Error: {e}")
            time.sleep(0.05)


#####################################################################
# LiPo utilities (unchanged)
#####################################################################
def moving_average_chunk(chunk, window_size=3, axis=0, mode='valid'):
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode=mode), axis, chunk)

def get_lipo_actions(actions, prev_chunk):
    """Blend CHUNK-sized windows with LiPo, return smoothed 7-DoF joint sequences."""
    smoothed = np.zeros_like(actions[:, :7])
    count = np.zeros(actions.shape[0])
    for start in range(0, actions.shape[0] - CHUNK + 1, 1):
        action_chunk = actions[start:start+CHUNK, :7]
        solved, _ = lipo.solve(
            action_chunk, prev_chunk,
            len_past_actions=BLEND if (prev_chunk is not None) else 0
        )
        chosen = solved if solved is not None else action_chunk
        span = CHUNK - BLEND
        for i in range(span):
            idx = start + i
            if idx < smoothed.shape[0]:
                smoothed[idx] += chosen[i]
                count[idx] += 1
        prev_chunk = chosen.copy()
    for i in range(smoothed.shape[0]):
        smoothed[i] = smoothed[i] / count[i] if count[i] > 0 else actions[i, :7]
    return smoothed


#####################################################################
# Execution Loop (Polymetis streaming)
#####################################################################
def execute_actions_loop(robot: RobotInterface, gripper:GripperInterface, fps=FPS):
    """
    Start a non-blocking joint impedance controller and stream targets at `fps`.
    We use update_desired_joint_positions(...) for smooth tracking.
    """
    dt = 1.0 / fps
    current_chunk = None
    current_gripper = None  # placeholder
    playhead = 0
    prev_tail = None  # LiPo past window

    # Start controller once (non-blocking)
    robot.start_joint_impedance(blocking=False)
    

    # Optional: let controller initialize
    time.sleep(0.05)

    # Gripper: Polymetis doesn't command gripper — keep your previous gripper bridge if needed.

    last_closed_time = None
    hold_duration = 5.0  # seconds

    grip_val = 0.04
    closed = False 
    
    while True:
        try:
            # Pull new chunk if available
            try:
                new_msg = action_queue.get_nowait()
                new_actions = np.array(new_msg["action.single_arm"])
                new_gripper = np.array(new_msg["action.gripper"])
                depth_mean = new_msg.get("depth_mean", 10.0)

                # Downsample & smooth (like your original)
                new_actions = new_actions[::2]
                new_gripper = new_gripper[::2]
                new_actions = moving_average_chunk(new_actions, window_size=2)

                if current_chunk is not None:
                    # Build tail for LiPo (last 1–2 samples around playhead)
                    tail = current_chunk[max(playhead-1, 0):playhead+1]
                    current_chunk = get_lipo_actions(new_actions, tail)
                    current_gripper = new_gripper
                    playhead = 0
                else:
                    current_chunk = new_actions
                    current_gripper = new_gripper
                    playhead = 0

            except queue.Empty:
                pass

            if current_chunk is None:
                time.sleep(dt)
                continue
            if playhead >= len(current_chunk):
                time.sleep(dt)
                continue

            # === Execute one step (stream target joints) ===
            q_cmd = current_chunk[playhead].astype(np.float32)
            # Safety: ensure 7-DoF vector
            if q_cmd.shape[0] != 7:
                print(f"[Executor] Bad action shape: {q_cmd.shape}")
                playhead += 1
                time.sleep(dt)
                continue

            # Send desired joint positions (non-blocking policy running)
            # robot.update_desired_joint_positions(torch.tensor(q_cmd, dtype=torch.float32))
            robot.move_to_joint_positions(torch.tensor(q_cmd, dtype=torch.float32))

            # (Optional) read back state occasionally
            # if playhead % 10 == 0:
            #     s = robot.get_robot_state()
            #     print("q_now[:2] =", s.joint_positions[:2])

            # Gripper logic placeholder (keep your original external gripper process)
            # e.g., if using an external gripper bridge: publish grip_val there.
            grip_val = float(current_gripper[playhead])

            if grip_val <= 0.02:
                gripper.goto(width=0.0, speed=1.00, force=30.0)
                closed = True
            if closed and grip_val >= 0.03:  
                gripper.goto(width=1.0, speed=1.00, force=30.0)
                closed = False

            playhead += 1
            time.sleep(dt)

        except Exception as e:
            print(f"[Executor] Error executing action: {e}")
            # Polymetis doesn’t expose recover here; you can restart controller if needed:
            try:
                robot.start_joint_impedance(blocking=False)
            except Exception:
                pass
            time.sleep(0.1)


#####################################################################
# Main
#####################################################################
def main(args: ArgsConfig):
    # Connect to Polymetis
    robot = RobotInterface(
        ip_address=args.poly_ip,
        port=args.poly_port,
        enforce_version=args.enforce_version
    )


    gripper = GripperInterface(
        ip_address="localhost",
    )

    # Quick ping
    s = robot.get_robot_state()
    print("[Polymetis] Connected. Current joints:", np.array(s.joint_positions, dtype=np.float32))

    # Optional: go home using server defaults
    if args.go_home_on_start:
        print("[Polymetis] Going home...")
        try:
            robot.go_home(blocking=True)
        except Exception as e:
            print(f"[Polymetis] go_home skipped: {e}")

    # Threads: fetch & execute
    if args.client:
        fetch_thread = threading.Thread(target=fetch_actions_loop, args=(args, robot, gripper), daemon=True)
        exec_thread = threading.Thread(target=execute_actions_loop, args=(robot, gripper, args.hz), daemon=True)
        fetch_thread.start()
        exec_thread.start()
        fetch_thread.join()
        exec_thread.join()
    else:
        raise ValueError("Client mode only.")

if __name__ == "__main__":
    cfg = ArgsConfig(
        host="192.168.3.20",   # GR00T inference server
        port=5001,
        client=True,
        poly_ip="localhost",   # Polymetis controller server
        poly_port=50051,
        enforce_version=False,
        go_home_on_start=False,
        hz=FPS
    )
    main(cfg)
