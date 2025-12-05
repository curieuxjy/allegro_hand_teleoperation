#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
GeoRT Allegro Deployer with Pinch Override

This module deploys GeoRT models to Allegro hands with pinch gesture detection.
When index_pinch_{left,right} topic is True, post_processing_commands() overrides
thumb and index finger joints to predefined pinch positions (Step D).
"""

import time
import math
import argparse
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool

from manus_mocap import ManusMocap
from allegro import AllegroCommandForwarder
from geort import load_model

class GeortAllegroDeployer(Node):
    def __init__(self, side: str, mocap: ManusMocap, model, loop_hz: float = 50.0,
                 smoothing_alpha: float = 0.9, two_robots: bool = True):
        super().__init__(f'geort_allegro_deployer_{side}')
        self.mocap = mocap
        self.model = model
        self.side = side

        # Create AllegroCommandForwarder with appropriate configuration
        # two_robots=True: dual robot system (uses _l/_r suffix)
        # two_robots=False: single robot system (no suffix)
        self.allegro = AllegroCommandForwarder(side=side, two_robots=two_robots)
        self.publisher_ = self.allegro.publisher_

        # Parameters (declarable/tweakable)
        self.declare_parameter('smoothing_alpha', float(smoothing_alpha))   # EMA alpha: 0..1 (1 = no smoothing)
        self.declare_parameter('max_delta', 0.05)        # rad per tick max change (0 disables)
        self.declare_parameter('snap_on_start', True)    # Whether to initialize to target directly on the first frame
        self.declare_parameter('loop_hz', float(loop_hz))

        self.alpha = float(self.get_parameter('smoothing_alpha').value)
        self.max_delta = float(self.get_parameter('max_delta').value)
        self.snap_on_start = bool(self.get_parameter('snap_on_start').value)

        print(f"[{side.capitalize()} Deployer] Smoothing alpha: {self.alpha}")

        # internal state
        self._last_cmd = None   # numpy array shape (16,)
        self._initialized = False

        # Index pinch state (subscribe to index_pinch topic)
        self._index_pinch = False
        pinch_topic = f'/index_pinch_{side}'
        self.create_subscription(Bool, pinch_topic, self._index_pinch_callback, 10)
        print(f"[{side.capitalize()} Deployer] Subscribing to {pinch_topic}")

        # Timer
        timer_period = 1.0 / float(self.get_parameter('loop_hz').value)
        self.create_timer(timer_period, self._on_timer)

        # Joint Limits
        self.allegro_dof_lower = [
            0.2630, -0.1050, -0.1890, -0.1620,   # Thumb
            -0.4700, -0.1960, -0.1740, -0.2270,  # Index
            -0.4700, -0.1960, -0.1740, -0.2270,  # Middle
            -0.4700, -0.1960, -0.1740, -0.2270,  # Ring
        ]
        self.allegro_dof_upper = [
            1.3960, 1.1630, 1.6440, 1.7190,      # Thumb
            0.4700, 1.6100, 1.7090, 1.6180,      # Index
            0.4700, 1.6100, 1.7090, 1.6180,      # Middle
            0.4700, 1.6100, 1.7090, 1.6180,      # Ring
        ]

    def _index_pinch_callback(self, msg: Bool):
        """Callback for index pinch detection topic."""
        self._index_pinch = msg.data

    def post_processing_commands(self, qpos):
        """
        Post-process model output: reorder joints and apply calibration tweaks.

        Args:
            qpos: Model output with 16 joint angles in model order
                  [Index(0:4), Middle(4:8), Ring(8:12), Thumb(12:16)]

        Returns:
            numpy array (16,) in Allegro hardware order
            [Thumb(0:4), Index(4:8), Middle(8:12), Ring(12:16)]
        """
        qpos = np.asarray(qpos).flatten()

        # ========================================================================
        # Step A: Reorder from model output to hardware order
        # ========================================================================
        # Model order: Index, Middle, Ring, Thumb
        # Hardware order: Thumb, Index, Middle, Ring
        try:
            allegro_hw = np.concatenate([
                qpos[12:16],  # Thumb (joints 0-3)
                qpos[0:4],    # Index (joints 4-7)
                qpos[4:8],    # Middle (joints 8-11)
                qpos[8:12]    # Ring (joints 12-15)
            ])
        except Exception as e:
            self.get_logger().error(f"{self.side} reordering error: {e}")
            return np.zeros(16, dtype=float)

        allegro_hw = allegro_hw.astype(float)

        # ========================================================================
        # Step B: Apply per-joint calibration adjustments (OPTIONAL - NOT RECOMMENDED)
        # ========================================================================
        # WARNING: These adjustments are hardware-specific calibration tweaks.
        # In most cases, you should skip this step and use the model output directly.
        # These values were tuned for a specific robot setup and may not generalize well.
        # Consider removing or modifying these only if you observe systematic errors
        # in your specific deployment.
        #
        # Joint numbering (hardware order):
        # Thumb:  [0]=MCP_Spread, [1]=MCP_Stretch, [2]=PIP, [3]=DIP
        # Index:  [4]=MCP_Spread, [5]=MCP_Stretch, [6]=PIP, [7]=DIP
        # Middle: [8]=MCP_Spread, [9]=MCP_Stretch, [10]=PIP, [11]=DIP
        # Ring:   [12]=MCP_Spread, [13]=MCP_Stretch, [14]=PIP, [15]=DIP

        # # --- Thumb adjustments ---
        ###################333#########
        # allegro_hw[0] = allegro_hw[0] * 1.3 # + (20.0 * math.pi / 180.0)  # MCP_Spread: scale + offset
        # allegro_hw[1] = allegro_hw[1] * 1.2                              # MCP_Stretch: scale only
        # allegro_hw[2] = allegro_hw[2] * 0.9
        ###################################
        # allegro_hw[3] = allegro_hw[3] - (10.0 * math.pi / 180.0)        # DIP: offset

        # # --- Index adjustments ---
        # allegro_hw[7] = 0.0  # DIP: fixed position

        # # --- Middle adjustments ---
        # allegro_hw[11] = 0.0  # DIP: fixed position

        # # --- Ring adjustments ---
        # allegro_hw[15] = 0.0  # DIP: fixed position

        # ========================================================================
        # Step C: Clip to joint limits
        # ========================================================================
        allegro_hw = np.clip(allegro_hw, self.allegro_dof_lower, self.allegro_dof_upper)

        # ========================================================================
        # Step D: Override index finger commands when pinch detected
        # ========================================================================
        # When index_pinch is True (index 4 and 8 distance < threshold),
        # force index finger joints (4-7) to zero
        if self._index_pinch:
            # thumb forcing
            if self.side == "right":
                allegro_hw[0:4] = [0.9293, 0.74515, 0.72979, 1.07432]
            else:  # left
                allegro_hw[0:4] = [0.9293, 0.74515, 0.72979, 1.07432]
            # index forcing
            if self.side == "right":
                allegro_hw[4:8] = [-0.2041, 0.74358, 1.5130, 0.4091]
            else:  # left
                allegro_hw[4:8] = [0.2041, 0.74358, 1.5130, 0.4091]

        return allegro_hw

    def _on_timer(self):
        data = self.mocap.get()
        if data is None:
            # no mocap frame available
            return

        points = data.get('result', None)
        if points is None:
            return

        # forward through model (protect against model errors)
        try:
            qpos = self.model.forward(points)
        except Exception as e:
            self.get_logger().error(f"{self.side} model forward failed: {e}")
            return

        target = self.post_processing_commands(qpos)  # numpy (16,)

        # initialize last_cmd on first valid frame
        if not self._initialized:
            if self.snap_on_start:
                self._last_cmd = target.copy()
            else:
                self._last_cmd = np.zeros_like(target)
            self._initialized = True
            # publish initial value immediately (avoid initial silence)
            msg = Float64MultiArray()
            msg.data = self._last_cmd.tolist()
            self.publisher_.publish(msg)
            self.get_logger().info(f"{self.side} initialized cmd: {np.round(self._last_cmd,3).tolist()}")
            return

        # EMA smoothing based on previous command
        alpha = self.alpha
        smoothed = alpha * target + (1.0 - alpha) * self._last_cmd

        # rate limit per joint
        if self.max_delta and self.max_delta > 0.0:
            delta = smoothed - self._last_cmd
            # clamp
            delta = np.clip(delta, -self.max_delta, self.max_delta)
            new_cmd = self._last_cmd + delta
        else:
            new_cmd = smoothed

        # publish
        msg = Float64MultiArray()
        msg.data = new_cmd.tolist()
        self.publisher_.publish(msg)

        # update internal state
        self._last_cmd = new_cmd

        # small debug for thumb only (in degrees for readability)
        thumb = new_cmd[:4]
        thumb_deg = np.array(thumb) * 180.0 / math.pi
        self.get_logger().debug(f"{self.side} Thumb(deg): {thumb_deg.round(1).tolist()}")

def main():
    parser = argparse.ArgumentParser(
        description='Deploy GeoRT models to Allegro hands with Manus gloves',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
  # Deploy both hands with trained checkpoints
  %(prog)s --right_ckpt "human1_right_1028_150817_allegro_right_last" \\
           --left_ckpt "human1_left_1028_150817_allegro_left_last"

  # Deploy with custom smoothing
  %(prog)s --right_ckpt "human1_right_1028_150817_allegro_right_last" \\
           --left_ckpt "human1_left_1028_150817_allegro_left_last" \\
           --smoothing_alpha 0.7
        ''')

    parser.add_argument('--right_ckpt', type=str, required=True,
                        help="Checkpoint tag for right hand model")
    parser.add_argument('--left_ckpt', type=str, required=True,
                        help="Checkpoint tag for left hand model")
    parser.add_argument('--loop_hz', type=float, default=100.0,
                        help="Control loop frequency in Hz (default: 100.0)")
    parser.add_argument('--smoothing_alpha', type=float, default=0.9,
                        help='EMA smoothing alpha (0..1, 1=no smoothing, default: 0.9)')
    parser.add_argument('--use_last', action='store_true',
                        help='Load last checkpoints instead of best checkpoints (default: best)')

    args = parser.parse_args()

    # Validate alpha value
    if not (0.0 <= args.smoothing_alpha <= 1.0):
        raise ValueError("smoothing_alpha must be between 0.0 and 1.0")

    rclpy.init()

    # mocap nodes
    right_mocap = ManusMocap(node_name='manus_mocap_right', topic_name='/manus_poses_right')
    left_mocap  = ManusMocap(node_name='manus_mocap_left',  topic_name='/manus_poses_left')

    # Load models from checkpoint tags
    epoch_to_load = 0 if args.use_last else 'best'
    print(f"[Loading] Right hand model: {args.right_ckpt} (checkpoint: {'last' if args.use_last else 'best'})")
    right_model = load_model(args.right_ckpt, epoch=epoch_to_load)
    print(f"[Loading] Left hand model: {args.left_ckpt} (checkpoint: {'last' if args.use_last else 'best'})")
    left_model = load_model(args.left_ckpt, epoch=epoch_to_load)

    # deployers with configurable parameters
    # two_robots=True: dual robot system (uses _l/_r suffix for controllers)
    right_deployer = GeortAllegroDeployer(
        "right", right_mocap, right_model,
        loop_hz=args.loop_hz,
        smoothing_alpha=args.smoothing_alpha,
        two_robots=True
    )
    left_deployer = GeortAllegroDeployer(
        "left", left_mocap, left_model,
        loop_hz=args.loop_hz,
        smoothing_alpha=args.smoothing_alpha,
        two_robots=True
    )

    executor = MultiThreadedExecutor()
    executor.add_node(right_mocap)
    executor.add_node(right_deployer)
    executor.add_node(left_mocap)
    executor.add_node(left_deployer)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        right_mocap.destroy_node()
        right_deployer.destroy_node()
        left_mocap.destroy_node()
        left_deployer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

    # After Calibration:
    # python glove_based/geort_allegro_deploy_pinch.py \
    #     --right_ckpt "miller_right_1203_145107_allegro_right_2025-12-03_16-27-37_s10" \
    #     --left_ckpt "miller_left_1203_150345_allegro_left_2025-12-03_16-13-42_s10"


    # ```bash
    # // 1
    # python glove_based/geort_allegro_deploy_pinch.py \
    #     --right_ckpt "miller_right_1120_101651_allegro_right_2025-11-20_11-04-40_s10" \
    #     --left_ckpt "miller_left_1120_102542_allegro_left_2025-11-20_11-05-32_s10"
    # // 2
    #     python glove_based/geort_allegro_deploy_pinch.py \
    #     --right_ckpt "miller_right_1120_104808_allegro_right_2025-11-20_12-00-43_s10" \
    #     --left_ckpt "miller_left_1120_103637_allegro_left_2025-11-20_12-00-45_s10"
    # //3
    # python glove_based/geort_allegro_deploy_pinch.py \
    #     --right_ckpt "miller_right_1120_104808_allegro_right_2025-11-20_12-29-03_s10" \
    #     --left_ckpt "miller_left_1120_103637_allegro_left_2025-11-20_12-29-12_s10"
    # // 4
    # python glove_based/geort_allegro_deploy_pinch.py \
    #     --right_ckpt "miller_right_1120_101651_allegro_right_2025-11-20_13-26-08_s10" \
    #     --left_ckpt "miller_left_1120_102542_allegro_left_2025-11-20_13-26-00_s10"
    # ```
