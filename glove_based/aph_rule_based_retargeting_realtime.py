#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
Manus Glove to APH Hand Rule-Based Retargeting — REAL-TIME ROS2 Output

Like aph_rule_based_retargeting.py but with live hardware transmission added:
every glove callback publishes the latest 16-D qpos as a Float64MultiArray to
`allegro_hand_position_effort_controller/commands` via the AphCommandForwarder
from aph.py. Sim viewer, tuning sliders, ergonomics viz, and motion-recording
are all kept; the tuner exposes the same Broadcast toggle as the V6 realtime
script (driven via `self.modbus` for interface parity with retargeting_utils
TuningSliders).

APH-specific bits:
  * hardware I/O wraps AphCommandForwarder (controller activation + publisher)
    rather than V6's pymodbus stack
  * sign flip at the publish boundary on 10 joints (Thumb-1 abduction +
    non-thumb flex) because our sim URDF axes are opposite to the hardware
    /joint_states `ah_joint*` convention used by the ros2_control controller.
    See AphRosBroadcaster.HARDWARE_SIGN_FLIP and hora_deploy.py for the
    convention reference. The thumb mechanical range matches; only signs
    differ.

CLI (hardware options on top of the sim-only CLI)
  --no-publish        opt out of hardware publishing (sim/tune only)
  --no-sign-flip      publish qpos as-is (sim convention) — useful for
                      verifying on a sim-style /joint_states bridge
  --publish-hz        default broadcast rate Hz (default 100)
"""

import sys
import argparse
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
import sapien.core as sapien
from std_msgs.msg import Float64MultiArray
from manus_ros2_msgs.msg import ManusGlove

try:
    from .geort.utils.config_utils import get_config
    from .geort.env.hand_debug import HandKinematicModel, HandViewerEnv
    from . import retargeting_utils as ru
    from .aph import AphCommandForwarder
    from .aph_rule_based_retargeting import (
        APH_LOWER_LIMITS, APH_UPPER_LIMITS,
        APH_FINGER_RGB, APH_FINGER_RGBA,
        APH_SCALE_LAYOUT, APH_OFFSET_LAYOUT, APH_REVERSED_SCALE_KEYS,
        _aph_link_classifier, ManusAphSimNode,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from geort.utils.config_utils import get_config
    from geort.env.hand_debug import HandKinematicModel, HandViewerEnv
    import retargeting_utils as ru
    from aph import AphCommandForwarder
    from aph_rule_based_retargeting import (
        APH_LOWER_LIMITS, APH_UPPER_LIMITS,
        APH_FINGER_RGB, APH_FINGER_RGBA,
        APH_SCALE_LAYOUT, APH_OFFSET_LAYOUT, APH_REVERSED_SCALE_KEYS,
        _aph_link_classifier, ManusAphSimNode,
    )


class AphRosBroadcaster:
    """ROS2-based broadcaster for APH hardware. Wraps an AphCommandForwarder
    (which handles controller activation + Float64MultiArray publishing) and
    exposes the same interface that retargeting_utils.TuningSliders expects
    from a Modbus broadcaster (start/stop/maybe_send + broadcasting flag +
    n_sent / send_errors / _bc_target_hz attrs).

    Sign-flip at the publish boundary:
      Our sim URDF (aph_right.urdf) declares the thumb abduction joint
      (joint_00) and all non-thumb flex joints (jN1..jN3) with NEGATIVE
      ranges, while the hardware `/joint_states` topic (ah_joint00..33
      from allegro_hand_position_effort_controller) uses POSITIVE values
      for both. See hora_deploy.py CANONICAL_AH for the reference values.
      We negate those 10 entries before publishing so the sim qpos drives
      the hardware to the matching physical pose.
    """

    # 16-D sim qpos indices (URDF order Thumb→Index→Middle→Ring) that need
    # sign flip before publishing to hardware. Spreads and thumb flex
    # (joints 01..03) are left as-is — they share sign with the hardware
    # convention.
    HARDWARE_SIGN_FLIP = (
        0,                    # Thumb-1 abduction (joint_00)
        5,  6,  7,            # Index   flex      (joint_11..13)
        9, 10, 11,            # Middle  flex      (joint_21..23)
        13, 14, 15,           # Ring    flex      (joint_31..33)
    )

    def __init__(self, default_hz: float = 100.0):
        self.forwarder = None
        self.connected = False
        self.broadcasting = False
        self.send_errors = 0
        self.n_sent = 0

        self._bc_t0 = None
        self._bc_target_hz = float(default_hz)
        self._bc_period = 1.0 / self._bc_target_hz
        self._bc_next_t = None
        self.last_stats = None

    # ----- connection ------------------------------------------------
    def connect(self) -> bool:
        """Create the AphCommandForwarder (activates the controller and
        creates the position-commands publisher). Returns False if either
        step failed; the script can still run sim/tune-only in that case."""
        if self.connected:
            return True
        try:
            self.forwarder = AphCommandForwarder()
        except Exception as e:
            print(f'[aph-broadcast] forwarder init failed: {e}')
            return False
        if self.forwarder.publisher_ is None:
            return False
        self.connected = True
        return True

    def disconnect(self):
        """Send the safe base position once, then destroy the forwarder
        node so rclpy can shut down cleanly."""
        if self.forwarder is not None:
            try:
                self.forwarder.return_to_base()
            except Exception:
                pass
            try:
                self.forwarder.destroy_node()
            except Exception:
                pass
        self.forwarder = None
        self.connected = False

    # ----- broadcasting ----------------------------------------------
    def start(self, target_hz: float = None):
        if target_hz is None:
            target_hz = self._bc_target_hz
        target_hz = float(max(0.1, min(target_hz, 120.0)))
        self._bc_target_hz = target_hz
        self._bc_period = 1.0 / target_hz
        self._bc_t0 = time.monotonic()
        self._bc_next_t = None
        self.n_sent = 0
        self.send_errors = 0
        print(f'[aph-broadcast] start @ {target_hz:.1f} Hz')
        self.broadcasting = True

    def stop(self):
        self.broadcasting = False
        elapsed = (time.monotonic() - self._bc_t0) if self._bc_t0 else 0.0
        fps = (self.n_sent / elapsed) if elapsed > 0 else 0.0
        self.last_stats = {
            'n': self.n_sent, 'elapsed': elapsed,
            'fps': fps, 'errors': self.send_errors,
        }
        return self.last_stats

    def _publish_qpos(self, qpos_rad) -> bool:
        if not self.connected or self.forwarder is None or \
                self.forwarder.publisher_ is None:
            self.send_errors += 1
            return False
        cmd = np.array(qpos_rad, dtype=np.float64).copy()
        for idx in self.HARDWARE_SIGN_FLIP:
            cmd[idx] = -cmd[idx]
        try:
            msg = Float64MultiArray()
            msg.data = cmd.tolist()
            self.forwarder.publisher_.publish(msg)
            return True
        except Exception:
            self.send_errors += 1
            return False

    def maybe_send(self, qpos_rad):
        """Drift-corrected scheduler — call on every glove callback. Same
        cadence pattern as realtime_utils.ModbusBroadcaster."""
        if not self.broadcasting:
            return
        now = time.monotonic()
        if self._bc_next_t is None:
            if self._publish_qpos(qpos_rad):
                self.n_sent += 1
            self._bc_next_t = now + self._bc_period
        elif now >= self._bc_next_t:
            if self._publish_qpos(qpos_rad):
                self.n_sent += 1
            self._bc_next_t += self._bc_period
            if now > self._bc_next_t:
                self._bc_next_t = now + self._bc_period


class ManusAphRealtimeNode(ManusAphSimNode):
    """ManusAphSimNode + a live `modbus` reference (named for interface
    parity with retargeting_utils.TuningSliders; here it actually holds an
    AphRosBroadcaster). Recording-to-txt and the entire glove parsing /
    transform pipeline are inherited unchanged from the sim node; this
    subclass only adds the hardware dispatch hook in glove_callback."""

    def __init__(self, side: str = 'right',
                 broadcaster: AphRosBroadcaster = None):
        # Re-init as a distinct ROS node name (mirrors v6 realtime pattern)
        # so introspection can tell sim and realtime variants apart.
        Node.__init__(self, 'manus_aph_realtime_node')
        self.side = side.lower()

        self.alpha = 0.2
        self.prev_arr = None

        self.scales = dict(self.DEFAULT_SCALES)
        self.offsets = dict(self.DEFAULT_OFFSETS)
        ru.load_tuning_file(self.TUNING_FILE, self.scales, self.offsets,
                            logger=self.get_logger())

        self.latest_qpos = None
        self.latest_glove16 = None
        self._got_first = False

        # The tuner reads `self.modbus` (see retargeting_utils.TuningSliders
        # broadcast callbacks). For APH that's actually our ROS broadcaster.
        self.modbus = broadcaster

        # Motion recording — independent of broadcast.
        self.recording = False
        self.recorded_frames = []
        self._rec_t0 = None
        self._rec_target_hz = 120.0
        self._rec_period = 1.0 / 120.0
        self._rec_next_t = None
        self.last_record_stats = None

        topic = f'/manus_glove_{self.side}'
        self.create_subscription(ManusGlove, topic, self.glove_callback, 20)
        self.get_logger().info(f'Subscribed to {topic}')

    def glove_callback(self, msg: ManusGlove):
        """Compute qpos (inherited transform), broadcast to hardware, and
        record."""
        if len(msg.ergonomics) == 0:
            return
        glove16 = self._ergonomics_to_array(msg.ergonomics)
        self.latest_glove16 = glove16
        self.latest_qpos = self.transform_glove_to_aph(glove16)

        if self.modbus is not None:
            self.modbus.maybe_send(self.latest_qpos)

        if self.recording:
            now = time.monotonic()
            if self._rec_next_t is None:
                self.recorded_frames.append(self.latest_qpos.copy())
                self._rec_next_t = now + self._rec_period
            elif now >= self._rec_next_t:
                self.recorded_frames.append(self.latest_qpos.copy())
                self._rec_next_t += self._rec_period
                if now > self._rec_next_t:
                    self._rec_next_t = now + self._rec_period

        if not self._got_first:
            self._got_first = True
            self.get_logger().info(
                f'First glove command received ({len(msg.ergonomics)} entries).'
            )


def main():
    parser = argparse.ArgumentParser(
        description='aph hand rule-based retargeting with real-time ROS2 output.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: sim + tuner + ROS2 publishing to
  # /allegro_hand_position_effort_controller/commands (broadcast starts OFF
  # — click Start Broadcast in the slider window when ready)
  python aph_rule_based_retargeting_realtime.py

  # Sim/tune only, no hardware output
  python aph_rule_based_retargeting_realtime.py --no-publish

  # Headless (no Sapien viewer), publish + tuner only
  python aph_rule_based_retargeting_realtime.py --no-sim
        """
    )
    parser.add_argument('--hand', type=str, default='aph_right')
    parser.add_argument('--side', type=str, choices=['left', 'right'], default='right')
    parser.add_argument('--no-kinematic', action='store_false', dest='kinematic',
                        help='Physics-mode PD drive instead of kinematic snap.')
    parser.add_argument('--no-ergo-viz', action='store_true',
                        help='Disable the ergo / joint-fraction Open3D window.')
    parser.add_argument('--no-tune', action='store_false', dest='tune',
                        help='Disable the tkinter slider window.')
    parser.add_argument('--no-sim', action='store_false', dest='sim',
                        help='Skip Sapien viewer (publish + tuner only).')

    # Hardware output options
    parser.add_argument('--no-publish', action='store_false', dest='publish_enabled',
                        help='Disable ROS2 hardware publishing entirely.')
    parser.add_argument('--publish-hz', type=float, default=100.0,
                        help='Default publish rate (Hz). Adjustable from the slider window.')

    args = parser.parse_args()

    # ROS init must happen before constructing the broadcaster's
    # AphCommandForwarder (which is itself a rclpy.Node and uses services
    # during activation).
    rclpy.init()

    # --------------------------------------------------------------------
    # Hardware broadcaster (soft-fail: sim/tune still works without it)
    # --------------------------------------------------------------------
    broadcaster = None
    if args.publish_enabled:
        try:
            broadcaster = AphRosBroadcaster(default_hz=args.publish_hz)
            if broadcaster.connect():
                print('[main] APH hardware publisher connected.')
            else:
                print('[main] APH hardware publisher: controller activation '
                      'failed — broadcast disabled. Sliders will still run; '
                      'check that allegro_hand_position_effort_controller is '
                      'loaded and re-Start.')
        except Exception as e:
            print(f'[main] hardware publisher init error: {e} — broadcast disabled.')
            broadcaster = None

    # --------------------------------------------------------------------
    # Sapien (optional)
    # --------------------------------------------------------------------
    model = None
    viewer_env = None
    if args.sim:
        engine, renderer, scene = ru.setup_sapien_scene()
        config = get_config(args.hand)
        print(f"[main] loading sim config: {config['name']}")
        model = HandKinematicModel.build_from_config(config, scene=scene, render=False)
        model.hand.set_root_pose(sapien.Pose([0.0, 0.0, 0.35], [0.695, 0.0, -0.718, 0.0]))
        ru.colorize_articulation(model.hand, _aph_link_classifier)
        if not args.kinematic:
            ru.tune_pd_drives(model)
        links = [info['link'] for info in config['fingertip_link']]
        offsets = [info['center_offset'] for info in config['fingertip_link']]
        model.initialize_keypoint(links, offsets)
        viewer_env = HandViewerEnv([model], scene=scene, renderer=renderer)

    # --------------------------------------------------------------------
    # Optional ergo viz
    # --------------------------------------------------------------------
    ergo_viz = None
    if not args.no_ergo_viz:
        ergo_viz = ru.ErgonomicsBarViz(
            APH_LOWER_LIMITS, APH_UPPER_LIMITS,
            fingers=['thumb', 'index', 'middle', 'ring'],
            finger_rgb=APH_FINGER_RGB,
            window_title='Manus ergonomics (front) / aph joint fraction (back)',
            window_size=(1000, 620),
        )

    # --------------------------------------------------------------------
    # ROS node + background spin
    # --------------------------------------------------------------------
    node = ManusAphRealtimeNode(side=args.side, broadcaster=broadcaster)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # --------------------------------------------------------------------
    # Optional tuner
    # --------------------------------------------------------------------
    tuner = None
    if args.tune:
        try:
            tuner = ru.TuningSliders(
                node,
                scale_layout=APH_SCALE_LAYOUT,
                offset_layout=APH_OFFSET_LAYOUT,
                reversed_scale_keys=APH_REVERSED_SCALE_KEYS,
                scale_magnitude_max=3.5, scale_res=0.1,
                offset_min=-90.0, offset_max=90.0, offset_res=5.0,
                slider_length=200,
                with_load=True,
                with_record=True, default_record_hz=50,
                with_broadcast=True, default_broadcast_hz=args.publish_hz,
                title='aph Retargeting Tuning (Realtime ROS2)',
            )
            print('[main] tuning slider window open.')
        except RuntimeError as e:
            print(f'[main] failed to open tuner: {e}')

    mode_str = 'kinematic snap' if args.kinematic else 'physics / PD'
    pub_str = ('publish ENABLED' if broadcaster is not None and broadcaster.connected
               else 'publish OFF')
    sim_str = 'sim ON' if model is not None else 'sim OFF'
    print(f"[main] running ({sim_str}, {mode_str}, {pub_str}). "
          f"Waiting for /manus_glove_{args.side} ...")

    try:
        while True:
            qpos = node.latest_qpos
            if qpos is not None and model is not None:
                ru.apply_qpos(model, qpos, args.kinematic)

            if viewer_env is not None:
                if args.kinematic:
                    viewer_env._update_tip_positions()
                    viewer_env._update_axis_markers()
                    viewer_env.scene.update_render()
                    viewer_env.viewer.render()
                else:
                    viewer_env.update()
            else:
                time.sleep(0.005)

            if ergo_viz is not None:
                ergo_viz.update(node.latest_glove16, node.latest_qpos)
                ergo_viz.render()

            if tuner is not None:
                tuner.poll()
    except KeyboardInterrupt:
        pass
    finally:
        if broadcaster is not None:
            if broadcaster.broadcasting:
                broadcaster.stop()
            broadcaster.disconnect()
        if tuner is not None:
            tuner.close()
        if ergo_viz is not None:
            ergo_viz.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
