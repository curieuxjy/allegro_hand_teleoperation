#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.
"""
Real-time GeoRT evaluation with optional motion recording.

Two modes:
  --kinematic        : snap qpos directly each frame (no physics). Used to
                       verify the IK output itself without collision
                       interference. Recorded qpos == IK prediction.
  default (physics)  : PD-drive joints toward IK target, run scene.step().
                       Recorded qpos is the qpos AFTER physics resolution,
                       so it reflects self-collision pushback / damping —
                       this is the "collision-aware" motion the user wanted
                       to capture for downstream replay or analysis.

Recording (--record):
  Buffers (time, human_points, qpos_target, qpos_actual) per frame and
  writes a .npz to glove_based/data/ on Ctrl+C (or normal shutdown).
  qpos_target = raw IK output; qpos_actual = post-physics qpos read back
  from sapien. Under --kinematic they are identical (snap is exact);
  under physics they may differ — that difference IS the value of this
  recording.

CLI:
  --hand v6_right.json
  --ckpt <substring>            (checkpoint tag)
  --kinematic                   (no physics; quick verification)
  --no_self_collision           (physics on, but intra-hand contacts filtered)
  --kp / --kd / --force_limit   (PD tuning for physics mode)
  --record [--record_name NAME] (save motion to data/<NAME>.npz)
"""
import argparse
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from manus_mocap import ManusMocap
from geort import load_model, get_config
from geort.env.hand import HandKinematicModel
from geort.env.hand_static import color_links
from geort.utils.path import get_data_root


# Re-uses the helper pattern from geort_replay_evaluation.py — filters out
# all intra-articulation contacts by sharing a single collision group whose
# mask excludes itself.
def disable_self_collision(articulation):
    GROUP = 0x0001
    MASK = 0xFFFE
    for link in articulation.get_links():
        try:
            shapes = link.get_collision_shapes()
        except AttributeError:
            shapes = []
        for shape in shapes:
            for sig in ((GROUP, MASK, 0, 0), (GROUP, MASK), (GROUP,)):
                try:
                    shape.set_collision_group(*sig)
                    break
                except (AttributeError, TypeError):
                    try:
                        shape.set_collision_groups(*sig)
                        break
                    except (AttributeError, TypeError):
                        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand', type=str, default='allegro')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--use_last', action='store_true',
                        help='Load last.pth instead of best.pth (default: best)')

    # Verification / physics modes
    parser.add_argument('--kinematic', action='store_true',
                        help='Snap qpos directly (no physics). Use to verify '
                             'pure IK output; recorded qpos == IK target.')
    parser.add_argument('--no_self_collision', action='store_true',
                        help='Physics on, but filter intra-articulation contacts. '
                             'Reduces divergence without losing collisions with '
                             'external objects (e.g. ground).')
    parser.add_argument('--kp', type=float, default=400.0,
                        help='PD position gain (default 400).')
    parser.add_argument('--kd', type=float, default=40.0,
                        help='PD velocity gain (default 40; near-critical for kp=400).')
    parser.add_argument('--force_limit', type=float, default=10.0,
                        help='Per-joint force/torque cap (default 10).')

    # Recording
    parser.add_argument('--record', action='store_true',
                        help='Record motion frames (time, human_points, qpos_target, '
                             'qpos_actual) to a .npz in glove_based/data/.')
    parser.add_argument('--record_name', type=str, default='',
                        help='Recording filename stem (default: realtime_<hand>_<time>).')

    args = parser.parse_args()

    # ROS2 init & mocap node
    rclpy.init()
    side = 'left' if 'left' in args.hand else 'right'
    mocap = ManusMocap(topic_name=f'/manus_poses_{side}')
    executor = MultiThreadedExecutor()
    executor.add_node(mocap)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # GeoRT model & hand simulator
    epoch_to_load = 0 if args.use_last else 'best'
    model = load_model(args.ckpt, epoch=epoch_to_load)

    config = get_config(args.hand)
    hand = HandKinematicModel.build_from_config(config, render=True)
    color_links(hand.get_renderer(), hand.hand)

    # PD gains + force limit — applied even under --kinematic (harmless there).
    for joint in hand.all_joints:
        joint.set_drive_property(args.kp, args.kd, force_limit=args.force_limit)

    if args.no_self_collision:
        disable_self_collision(hand.hand)
        print('[realtime] self-collision disabled')

    viewer_env = hand.get_viewer_env()

    # Joint limits for kinematic clipping
    lo, hi = hand.get_joint_limit()
    lo = np.asarray(lo)
    hi = np.asarray(hi)

    recorded = []  # list of dicts; flushed to .npz on shutdown

    print(f'[realtime] mode={"kinematic" if args.kinematic else "physics"}, '
          f'record={args.record}')

    try:
        while rclpy.ok():
            if args.kinematic:
                # No physics step — render only. Pose on screen is the exact
                # IK prediction.
                viewer_env.scene.update_render()
                viewer_env.viewer.render()
            else:
                for _ in range(10):
                    viewer_env.update()

            data = mocap.get()
            if data['status'] != 'streaming' or data['result'] is None:
                continue
            points = data['result']

            # IK forward
            t1 = time.perf_counter()
            qpos_target = model.forward(points)
            t2 = time.perf_counter()

            # Drive (or snap)
            if args.kinematic:
                qpos_clipped = np.clip(qpos_target, lo + 1e-3, hi - 1e-3)
                qpos_sim = hand.convert_user_order_to_sim_order(qpos_clipped)
                hand.hand.set_qpos(qpos_sim)
                hand.hand.set_qvel(np.zeros_like(qpos_sim))
            else:
                hand.set_qpos_target(qpos_target)

            # Read back actual qpos (post-physics or post-snap), in USER order
            # so it lines up with qpos_target.
            qpos_actual_sim = hand.hand.get_qpos()
            qpos_actual = hand.convert_sim_order_to_user_order(
                np.asarray(qpos_actual_sim))

            t3 = time.perf_counter()

            if args.record:
                recorded.append({
                    't': time.time(),
                    'human_points': np.asarray(points, dtype=np.float32).copy(),
                    'qpos_target': np.asarray(qpos_target, dtype=np.float32).copy(),
                    'qpos_actual': np.asarray(qpos_actual, dtype=np.float32).copy(),
                })

            # Lightweight timing log every ~30 frames to avoid log spam
            if len(recorded) % 30 == 1:
                print(f'IK {(t2-t1)*1000:.1f}ms  Sim {(t3-t2)*1000:.1f}ms  '
                      f'recorded={len(recorded)}')

    except KeyboardInterrupt:
        print('\n[realtime] interrupted')
    finally:
        # Flush recording to disk before shutting ROS down.
        if args.record and recorded:
            hand_stem = args.hand.replace('.json', '')
            time_tag = datetime.now().strftime('%m%d_%H%M%S')
            stem = args.record_name or f'realtime_{hand_stem}_{time_tag}'
            out_dir = Path(get_data_root())
            os.makedirs(out_dir, exist_ok=True)
            out_path = out_dir / f'{stem}.npz'
            np.savez(
                out_path,
                t=np.array([r['t'] for r in recorded], dtype=np.float64),
                human_points=np.stack([r['human_points'] for r in recorded]),
                qpos_target=np.stack([r['qpos_target'] for r in recorded]),
                qpos_actual=np.stack([r['qpos_actual'] for r in recorded]),
                meta=np.array([{
                    'hand': args.hand,
                    'ckpt': args.ckpt,
                    'kinematic': args.kinematic,
                    'no_self_collision': args.no_self_collision,
                    'kp': args.kp, 'kd': args.kd, 'force_limit': args.force_limit,
                    'n_frames': len(recorded),
                }], dtype=object),
            )
            print(f'[realtime] saved {len(recorded)} frames to {out_path}')

        executor.shutdown()
        mocap.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=0.1)


if __name__ == '__main__':
    main()
