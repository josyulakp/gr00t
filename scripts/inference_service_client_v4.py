"""
GR00T Inference Service - Socket Client
Fetches action chunks from custom GR00T inference server,
executes on Franka robot with blending between chunks.
"""

import time
import threading
import queue
from dataclasses import dataclass
import numpy as np
import cv2
import franky
import socket
import pickle
import pyrealsense2 as rs
from action_lipo import ActionLiPo


#####################################################################
# Robot Setup
#####################################################################
robot = franky.Robot("172.16.0.2")
robot.relative_dynamics_factor = 0.03
robot.set_collision_behavior([15.0]*7, [30.0]*6)

gripper = franky.Gripper("172.16.0.2")
max_gripper_width = 0.08  # ~80 mm

FPS = 30
chunk = 20
blend = 10
time_delay = 3
lipo = ActionLiPo(chunk_size=chunk, blending_horizon=blend, len_time_delay=time_delay)


#####################################################################
# Args
#####################################################################
@dataclass
class ArgsConfig:
    host: str = "localhost"   # server IP
    port: int = 5000          # server port
    client: bool = True


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
# Camera Setup (RealSense)
#####################################################################
def create_rs_camera(serial, width=640, height=480, fps=30):
    """Initialize and return a RealSense pipeline bound to a specific serial, with depth stream."""
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
    depth_scale = depth_sensor.get_depth_scale()
    return depth_scale


#####################################################################
# Queues
#####################################################################
action_queue = queue.Queue(maxsize=1)


#####################################################################
# Fetch Loop
#####################################################################
def fetch_actions_loop(args):
    """Continuously fetch actions from server with depth capture."""
    serial_left = "142422250807"
    serial_top = "025522060843"
    serial_wrist = "218622278163"

    rs_left, align_left = create_rs_camera(serial_left)
    rs_top, align_top = create_rs_camera(serial_top)
    rs_wrist, align_wrist = create_rs_camera(serial_wrist)

    depth_scale = get_depth_scale(rs_wrist)

    while True:
        try:
            # Grab frames
            left_frames = rs_left.wait_for_frames()
            top_frames  = rs_top.wait_for_frames()
            wrist_frames = rs_wrist.wait_for_frames()

            # Align
            left_frames = align_left.process(left_frames)
            top_frames  = align_top.process(top_frames)
            wrist_frames = align_wrist.process(wrist_frames)

            left_img  = np.asanyarray(left_frames.get_color_frame().get_data())
            top_img   = np.asanyarray(top_frames.get_color_frame().get_data())
            wrist_img = np.asanyarray(wrist_frames.get_color_frame().get_data())
            wrist_depth = np.asanyarray(wrist_frames.get_depth_frame().get_data())

            if left_img is None or top_img is None or wrist_img is None:
                continue

            # Robot state
            robot_state   = robot.current_joint_state.position.reshape(1, 7)
            gripper_state = np.array(gripper.width / 2.0).reshape(1, 1)

            # Observation
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
            
            cv2.imwrite("left_image.png", left_img)
            cv2.imwrite("top_image.png", top_img)
            cv2.imwrite("wrist_image.png", wrist_img)

            # Call server
            actions = gr00t_socket_client_call(obs, args.host, args.port)

            # Add depth heuristic
            depth_mean = wrist_depth.mean() * depth_scale
            actions["depth_mean"] = depth_mean

            # Replace queue contents
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
# Action Execution
#####################################################################
def moving_average_chunk(chunk, window_size=3, axis=0, mode='valid'):
    """Smooth chunk with moving average filter."""
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode=mode), axis, chunk)


def get_lipo_actions(actions, prev_chunk):
    smoothed_actions = np.zeros_like(actions[:, :7])
    count = np.zeros(actions.shape[0])
    for start in range(0, actions.shape[0] - chunk + 1, 1):
        action_chunk = actions[start:start+chunk, :7]
        solved, _ = lipo.solve(action_chunk, prev_chunk,
                               len_past_actions=blend if prev_chunk is not None else 0)
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

########################################################################
# Temporal Aggregation Utilities
########################################################################

# Global buffer for temporal aggregation
MAX_STEPS = 1000       # adjust based on rollout length
NUM_QUERIES = 8        # how many steps ahead the policy predicts
STATE_DIM = 7          # number of joints (Franka has 7)

# buffer: shape [MAX_STEPS, MAX_STEPS+NUM_QUERIES, STATE_DIM]
all_time_actions = np.zeros((MAX_STEPS, MAX_STEPS + NUM_QUERIES, STATE_DIM))

def temporal_aggregate(new_actions: np.ndarray, t: int) -> np.ndarray:
    """
    Insert new action predictions into buffer and aggregate actions for step t.

    Args:
        new_actions: np.ndarray [NUM_QUERIES, STATE_DIM]
        t: current timestep index

    Returns:
        Aggregated action for step t (STATE_DIM,)
    """
    global all_time_actions

    # Insert predictions into buffer (predicts t...t+NUM_QUERIES-1)
    horizon = min(NUM_QUERIES, new_actions.shape[0])
    all_time_actions[t, t:t+horizon, :] = new_actions[:horizon]

    # Extract all predictions from history that target step t
    actions_for_t = all_time_actions[:, t, :]
    mask = ~(actions_for_t == 0).all(axis=1)
    actions_for_t = actions_for_t[mask]

    if actions_for_t.shape[0] == 0:
        # Fallback to first prediction if no history
        return new_actions[0]

    # Exponential weights (fresh predictions weigh more)
    k = 0.01
    weights = np.exp(-k * np.arange(len(actions_for_t)))
    weights /= weights.sum()

    agg = (actions_for_t * weights[:, None]).sum(axis=0)
    return agg


########################################################################
# Executor with Temporal Aggregation
########################################################################

def execute_actions_loop(fps=30):
    """Execute actions with temporal aggregation + gripper handling."""
    dt = 1.0 / fps
    current_chunk = None
    current_gripper = None
    playhead = 0

    last_closed_time = None
    hold_duration = 5.0  # seconds to hold after closing

    # Ensure gripper is open at start
    if gripper.width / 2 <= 0.03:
        gripper.open(1.0)

    t_global = 0  # global step index (for temporal agg)

    while True:
        try:
            # === Get next action chunk from queue ===
            try:
                new_chunk = action_queue.get_nowait()
                new_actions = np.array(new_chunk["action.single_arm"])
                new_gripper = np.array(new_chunk["action.gripper"])
                depth_mean = new_chunk.get("depth_mean", 10.0)

                print("gripper ", new_gripper)

                # Downsample for speed
                new_actions = new_actions[::2]
                new_gripper = new_gripper[::2]

                # Reset playhead on new chunk
                current_chunk = new_actions
                current_gripper = new_gripper
                playhead = 0

            except queue.Empty:
                pass

            # Skip if nothing to execute
            if current_chunk is None or current_gripper is None:
                time.sleep(dt)
                continue
            if playhead >= len(current_chunk):
                time.sleep(dt)
                continue

            # === Temporal Aggregation ===
            action_pred_seq = current_chunk[playhead:]
            action = temporal_aggregate(action_pred_seq, t_global)

            # Execute smoothed joint action
            robot.move(franky.JointMotion(action.tolist()), asynchronous=True)

            # Gripper logic
            grip_val = current_gripper[min(playhead, len(current_gripper) - 1)]
            now = time.time()

            if grip_val < 0.02 and last_closed_time is None:
                print(f"[Gripper] CLOSE action detected at playhead={playhead}")

                # Hold pose briefly
                time.sleep(0.5)

                # Close gripper
                gripper.grasp_async(0.0, 1.0, 20.0, epsilon_outer=1.0)
                last_closed_time = now
                print("[Gripper] CLOSE triggered")

            if last_closed_time is not None and (now - last_closed_time) >= hold_duration:
                gripper.open(1.0)
                print("[Gripper] OPEN triggered after hold")
                last_closed_time = None

            playhead += 1
            t_global += 1
            time.sleep(dt)

        except Exception as e:
            print(f"[Executor] Error executing action: {e}")
            if robot.has_errors:
                robot.recover_from_errors()
            time.sleep(0.1)

#####################################################################
# Main
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
        raise ValueError("Client mode only.")


if __name__ == "__main__":
    config = ArgsConfig(host="192.168.3.20", port=5001, client=True)
    main(config)
