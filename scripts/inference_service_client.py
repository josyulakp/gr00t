# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
GR00T Inference Service

This script provides both ZMQ and HTTP server/client implementations for deploying GR00T models.
The HTTP server exposes a REST API for easy integration with web applications and other services.

1. Default is zmq server.

Run server: python scripts/inference_service.py --server
Run client: python scripts/inference_service.py --client

2. Run as Http Server:

Dependencies for `http_server` mode:
    => Server (runs GR00T model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

HTTP Server Usage:
    python scripts/inference_service.py --server --http-server --port 8000

HTTP Client Usage (assuming a server running on 0.0.0.0:8000):
    python scripts/inference_service.py --client --http-server --host 0.0.0.0 --port 8000

You can use bore to forward the port to your client: `159.223.171.199` is bore.pub.
    bore local 8000 --to 159.223.171.199
"""

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

###client
import socket
import pickle
import cv2
import numpy as np
import franky
from opencv import OpenCVCameraConfig, OpenCVCamera
import time
from franky import Affine, CartesianMotion, ReferenceType
from scipy.spatial.transform import Rotation as R

from action_lipo import ActionLiPo
from real_time_chunking import realtime_action

chunk = 50
blend = 10
time_delay = 3
lipo = ActionLiPo(chunk_size=chunk, blending_horizon=blend, len_time_delay=time_delay)



robot = franky.Robot("172.16.0.2")
robot.relative_dynamics_factor = 0.05
gripper = franky.Gripper("172.16.0.2")
max_gripper_width = 0.08  # ~80 mm
FPS = 30


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

def get_robot_state():
    pose = robot.current_pose
    gr_width = gripper.width / max_gripper_width
    if pose is not None:
        pos = pose.end_effector_pose.translation.tolist()
        ori = pose.end_effector_pose.quaternion
        return np.concatenate([pos, ori, [gr_width]])
    else:
        print("Robot pose is None")
        return None

@dataclass
class ArgsConfig:
    """Command line arguments for the inference service."""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """The embodiment tag for the model."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist"
    """The name of the data config to use."""

    port: int = 5555
    """The port number for the server."""

    host: str = "localhost"
    """The host address for the server."""

    server: bool = False
    """Whether to run the server."""

    client: bool = False
    """Whether to run the client."""

    denoising_steps: int = 4
    """The number of denoising steps to use."""

    api_token: str = None
    """API token for authentication. If not provided, authentication is disabled."""

    http_server: bool = False
    """Whether to run it as HTTP server. Default is ZMQ server."""


#####################################################################################


def _example_zmq_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example ZMQ client call to the server.
    """
    # Original ZMQ client mode
    # Create a policy wrapper
    policy_client = RobotInferenceClient(host=host, port=port, api_token=api_token)

    print("Available modality config available:")
    modality_configs = policy_client.get_modality_config()
    print(modality_configs.keys())

    time_start = time.time()
    action = policy_client.get_action(obs)
    print(f"Total time taken to get action from server: {time.time() - time_start} seconds")
    return action


def _example_http_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example HTTP client call to the server.
    """
    import json_numpy

    json_numpy.patch()
    import requests

    # Send request to HTTP server
    print("Testing HTTP server...")

    time_start = time.time()
    response = requests.post(f"http://{host}:{port}/act", json={"observation": obs})
    print(f"Total time taken to get action from HTTP server: {time.time() - time_start} seconds")

    if response.status_code == 200:
        action = response.json()
        return action
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}


def main(args: ArgsConfig):
    if args.server:
        # Create a policy
        # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
        # the model path, transform name, embodiment tag, and denoising steps for the robot
        # inference system. This policy object is then utilized in the server mode to start
        # the Robot Inference Server for making predictions based on the specified model and
        # configuration.

        # we will use an existing data config to create the modality config and transform
        # if a new data config is specified, this expect user to
        # construct your own modality config and transform
        # see gr00t/utils/data.py for more details
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

        # Start the server
        if args.http_server:
            from gr00t.eval.http_server import HTTPInferenceServer  # noqa: F401

            server = HTTPInferenceServer(
                policy, port=args.port, host=args.host, api_token=args.api_token
            )
            server.run()
        else:
            server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token)
            server.run()

    # Here is mainly a testing code
    elif args.client:
        prev_gripper = None
        while True:
            print("CALLED CLIENT")
            # In this mode, we will send a random observation to the server and get an action back
            # This is useful for testing the server and client connection
            top_img = cameras["right"].async_read()
            wrist_img = cameras["wrist"].async_read()
            left_img = cameras["left"].async_read()
            print(f"images shape {top_img.shape}{wrist_img.shape}{left_img.shape}")
            robot_state = robot.current_joint_state.position.reshape(1,7)
            gripper_state = np.array(gripper.width/2.0).reshape(1,1)


            # Making prediction...
            # - obs: video.ego_view: (1, 256, 256, 3)
            # - obs: state.left_arm: (1, 7)
            # - obs: state.right_arm: (1, 7)
            # - obs: state.left_hand: (1, 6)
            # - obs: state.right_hand: (1, 6)
            # - obs: state.waist: (1, 3)

            # - action: action.left_arm: (16, 7)
            # - action: action.right_arm: (16, 7)
            # - action: action.left_hand: (16, 6)
            # - action: action.right_hand: (16, 6)
            # - action: action.waist: (16, 3)
            cv2.imwrite("/home/prnuc/Documents/josyulak/gr00t/scripts/top_image.png", top_img[:, :, ::-1])
            cv2.imwrite("/home/prnuc/Documents/josyulak/gr00t/scripts/wrist_image.png", wrist_img[:, :, ::-1])
            cv2.imwrite("/home/prnuc/Documents/josyulak/gr00t/scripts/left_image.png", left_img[:, :, ::-1])

            obs = {
                "video.left":  left_img[:, :, ::-1].reshape(1,480,640,3), #np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
                "video.right": top_img[:, :, ::-1].reshape(1,480,640,3), #np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
                "video.wrist": wrist_img[:, :, ::-1].reshape(1,480,640,3), #np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
                "state.single_arm": robot_state, #np.random.rand(1,7) 
                "state.gripper": gripper_state, #np.random.rand(1,1)
                "annotation.human.action.task_description": ["Pick the bowl and place it in the green square"],
            }

            if args.http_server:
                action = _example_http_client_call(obs, args.host, args.port, args.api_token)
            else:
                action = _example_zmq_client_call(obs, args.host, args.port, args.api_token)

            ## realtime action chunking 
           
            
            for key, value in action.items():
                print(f"Action: {key}: {value}")
            # import ipdb; ipdb.set_trace()
            m_jp = franky.JointMotion(action["action.single_arm"][0].tolist())
            robot.move(m_jp)
            if prev_gripper is not None:
                if (action["action.gripper"] <=0.01).any() and (prev_gripper <= 0.01).any():
                    gripper.grasp_async(0.0, 1.0, 20.0, epsilon_outer=1.0)
                else: 
                    gripper.open_async(1.0)
            prev_gripper = action["action.gripper"]

    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
