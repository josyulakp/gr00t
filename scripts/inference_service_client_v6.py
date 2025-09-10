#!/usr/bin/env python3
"""
GR00T Inference Service + UMI Controllers
Streams EE poses (Cartesian) & gripper commands from GR00T inference
and executes on Franka using UMI's FrankaInterpolationController + WSGController.
"""

import os
import sys
import time
import socket
import pickle
import threading
import queue
import numpy as np
import click
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager


sys.path.insert(0, "/home/prnuc/Documents/josyulak/universal_manipulation_interface/")


# UMI imports
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.wsg_controller import WSGController
from umi.common.precise_sleep import precise_wait
from umi.real_world.keystroke_counter import KeystrokeCounter, KeyCode

# Optional: RealSense for observations
import pyrealsense2 as rs

from polymetis import RobotInterface

# LiPo blending (unchanged)
from action_lipo import ActionLiPo

from polymetis import GripperInterface



robot = RobotInterface(
    ip_address="localhost",
    port="50051",
    enforce_version=False
)


gripper = GripperInterface(
    ip_address="localhost",
)

#####################################################################
# Config
#####################################################################
FPS = 30
CHUNK = 20
BLEND = 10
TIME_DELAY = 3
QUEUE = queue.Queue(maxsize=1)   # where GR00T fetcher puts actions

#####################################################################
# GR00T socket client
#####################################################################
def gr00t_socket_client_call(obs: dict, host: str, port: int):
    data = pickle.dumps(obs) + b"<END>"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(data)

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
    return pickle.loads(response)

#####################################################################
# Camera helpers (optional, for obs to GR00T)
#####################################################################
def create_rs_camera(serial, width=640, height=480, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align

#####################################################################
# Fetch loop – send obs, receive EE pose chunks
#####################################################################
def fetch_actions_loop(host, port, stop_flag):
    """Fetch EE pose + gripper sequences from GR00T server and push to QUEUE."""
    serial_left = "142422250807"
    serial_top = "025522060843"
    serial_wrist = "218622278163"

    rs_left, align_left = create_rs_camera(serial_left, fps=FPS)
    rs_top, align_top = create_rs_camera(serial_top, fps=FPS)
    rs_wrist, align_wrist = create_rs_camera(serial_wrist, fps=FPS)

    while not stop_flag["stop"]:
        try:
            left_frames = align_left.process(rs_left.wait_for_frames())
            top_frames = align_top.process(rs_top.wait_for_frames())
            wrist_frames = align_wrist.process(rs_wrist.wait_for_frames())

            left_img = np.asanyarray(left_frames.get_color_frame().get_data())
            top_img = np.asanyarray(top_frames.get_color_frame().get_data())
            wrist_img = np.asanyarray(wrist_frames.get_color_frame().get_data())

            # Robot state from Polymetis
            state = robot.get_robot_state()
            q = np.array(state.joint_positions, dtype=np.float32).reshape(1, 7)

            # NOTE: Polymetis doesn't control the Franka gripper directly; keep your external
            # gripper bridge if needed. Here we pass a placeholder scalar.
            gripper_state = gripper.get_state()
            gripper_width_half = np.array([gripper_state.width], dtype=np.float32).reshape(1, 1)


            obs = {
                "video.left":  left_img.reshape(1, 480, 640, 3),
                "video.right": top_img.reshape(1, 480, 640, 3),
                "video.wrist": wrist_img.reshape(1, 480, 640, 3),
                "state.single_arm": q,
                "state.gripper": gripper_width_half,
                "annotation.human.action.task_description": ["Pick the bowl and place it in the green square"],
            }

            actions = gr00t_socket_client_call(obs, host, port)
            # Expecting:
            # actions["action.ee_pose"]: (T,6) array [x,y,z,rotvec]
            # actions["action.gripper"]: (T,) array

            if not QUEUE.empty():
                try: QUEUE.get_nowait()
                except queue.Empty: pass
            QUEUE.put_nowait(actions)

        except Exception as e:
            print(f"[Fetcher] Error: {e}")
            time.sleep(0.05)

#####################################################################
# Execution loop – schedule GR00T poses with UMI
#####################################################################
def execute_loop(controller: FrankaInterpolationController,
                 gripper: WSGController,
                 stop_flag, frequency=30, gripper_speed=200.0):

    dt = 1.0 / frequency
    t_start = time.monotonic()
    iter_idx = 0
    target_pose = controller.get_state()["ActualTCPPose"]
    gripper_target = gripper.get_state()["gripper_position"]
    print("target pose ", target_pose)
    while not stop_flag["stop"]:
        t_cycle_end = t_start + (iter_idx + 1) * dt
        t_sample = t_cycle_end - dt/2
        t_command_target = t_cycle_end + dt

        try:
            # Get new chunk if available
            try:
                msg = QUEUE.get_nowait()
                ee_chunk = np.array(msg["action.single_arm"])   # [T,6]
                grip_chunk = np.array(msg["action.gripper"]) # [T,]
                # Here we just take the first in chunk each cycle
                pose = ee_chunk[0]
                print("recieved pose ", pose)
                grip_val = grip_chunk[0]
                print("gripper val ", grip_val)
                # Convert orientation if needed (assume rotvec)
                pos = pose[:3]
                quat = pose[3:]  # qw, qx, qy, qz
                rotvec = st.Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_rotvec()

                target_pose[:3] = pos
                target_pose[3:] = rotvec
                gripper_target = float(np.clip(grip_val * 1000, 0, 90))  # mm scale

            except queue.Empty:
                pass

            # Schedule pose and gripper
            controller.schedule_waypoint(
                target_pose,
                t_command_target - time.monotonic() + time.time()
            )
            gripper.schedule_waypoint(
                gripper_target,
                t_command_target - time.monotonic() + time.time()
            )

            precise_wait(t_cycle_end)
            iter_idx += 1

        except Exception as e:
            print(f"[Executor] Error: {e}")
            time.sleep(0.1)

#####################################################################
# Main
#####################################################################
@click.command()
@click.option('--robot_ip', default="172.16.0.2")
@click.option('--gripper_ip', default="172.16.0.2")
@click.option('--gripper_port', default=1000, type=int)
@click.option('--host', default="192.168.3.20")
@click.option('--port', default=5001, type=int)
@click.option('--frequency', default=30, type=int)
def main(robot_ip, gripper_ip, gripper_port, host, port, frequency):
    stop_flag = {"stop": False}

    with SharedMemoryManager() as shm_manager, \
         WSGController(shm_manager, hostname=gripper_ip, port=gripper_port,
                       frequency=frequency, move_max_speed=400.0) as gripper, \
         FrankaInterpolationController(shm_manager, robot_ip=robot_ip,
                                       frequency=100, Kx_scale=5.0, Kxd_scale=2.0) as controller, \
         KeystrokeCounter() as key_counter:

        print("[System] Ready. Press 'q' to quit.")

        fetch_thread = threading.Thread(target=fetch_actions_loop, args=(host, port, stop_flag), daemon=True)
        exec_thread = threading.Thread(target=execute_loop, args=(controller, gripper, stop_flag, frequency), daemon=True)
        fetch_thread.start()
        exec_thread.start()

        while True:
            presses = key_counter.get_press_events()
            for k in presses:
                if k == KeyCode(char="q"):
                    stop_flag["stop"] = True
                    print("[System] Stop signal received.")
                    fetch_thread.join()
                    exec_thread.join()
                    return
            time.sleep(0.05)

if __name__ == "__main__":
    main()
