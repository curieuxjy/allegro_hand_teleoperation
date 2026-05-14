#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
Manus Glove to V6 Hand Rule-Based Retargeting — REAL-TIME Modbus Output

Like v6_rule_based_retargeting.py but with live Modbus transmission added:
every glove callback packages the latest 20-D qpos as encoder counts and
writes them to consecutive holding registers via pymodbus. Sim viewer,
tuning sliders, ergonomics viz, and motion-recording-to-txt are all kept;
the tuner exposes a separate Broadcast toggle for the Modbus stream.

Hardware/protocol bits live in realtime_utils.ModbusBroadcaster; the rest
of the shared UI/Sapien code lives in retargeting_utils. This file only
holds the V6-specific retargeting math + slider layout.

CLI (Modbus options on top of the sim-only CLI)
  --no-modbus           opt out of Modbus output (sim/tune only)
  --modbus-method       'rtu' (serial) or 'tcp'   (default rtu)
  --modbus-port         serial device path        (default /dev/ttyUSB0)
  --modbus-baud         serial baud rate          (default 2_000_000)
  --modbus-host         TCP host                  (default localhost)
  --modbus-tcp-port     TCP port                  (default 502)
  --modbus-slave-id     Modbus slave id           (default 1, auto-probed)
  --modbus-start-reg    starting holding register (default 0x10)
  --modbus-hz           default send rate Hz      (default 100)
"""

import sys
import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
import sapien.core as sapien
from manus_ros2_msgs.msg import ManusGlove

try:
    from .geort.utils.config_utils import get_config
    from .geort.env.hand_debug import HandKinematicModel, HandViewerEnv
    from . import retargeting_utils as ru
    from . import realtime_utils as rtu
    from .v6_rule_based_retargeting import (
        V6_LOWER_LIMITS, V6_UPPER_LIMITS,
        V6_FINGER_RGB, V6_FINGER_RGBA,
        V6_SCALE_LAYOUT, V6_OFFSET_LAYOUT, V6_REVERSED_SCALE_KEYS,
        _v6_link_classifier, ManusV6SimNode,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from geort.utils.config_utils import get_config
    from geort.env.hand_debug import HandKinematicModel, HandViewerEnv
    import retargeting_utils as ru
    import realtime_utils as rtu
    from v6_rule_based_retargeting import (
        V6_LOWER_LIMITS, V6_UPPER_LIMITS,
        V6_FINGER_RGB, V6_FINGER_RGBA,
        V6_SCALE_LAYOUT, V6_OFFSET_LAYOUT, V6_REVERSED_SCALE_KEYS,
        _v6_link_classifier, ManusV6SimNode,
    )


class ManusV6RealtimeNode(ManusV6SimNode):
    """ManusV6SimNode + a live `modbus` reference whose `maybe_send` is
    called from the glove callback. Recording-to-txt and the entire glove
    parsing / transform pipeline are inherited unchanged from the sim node;
    this subclass only adds the Modbus dispatch hook."""

    def __init__(self, side: str = 'right',
                 modbus: rtu.ModbusBroadcaster = None):
        # Rename the underlying node so ROS introspection distinguishes it
        # from the sim-only variant. Done by re-init'ing rclpy.Node directly
        # rather than super().__init__('manus_v6_sim_node').
        Node.__init__(self, 'manus_v6_realtime_node')
        self.side = side.lower()

        self.alpha = 0.2
        self.prev_arr = None

        self.scales = dict(self.DEFAULT_SCALES)
        self.offsets = dict(self.DEFAULT_OFFSETS)
        ru.load_tuning_file(self.TUNING_FILE, self.scales, self.offsets,
                            logger=self.get_logger())

        self.latest_qpos = None
        self.latest_glove20 = None
        self._got_first = False

        # Live Modbus output (None if --no-modbus or pymodbus missing).
        self.modbus = modbus

        # Motion recording — independent of Modbus broadcast.
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
        """Compute qpos (inherited transform), Modbus-broadcast, and
        record."""
        if len(msg.ergonomics) == 0:
            return
        glove20 = self._ergonomics_to_array(msg.ergonomics)
        self.latest_glove20 = glove20
        self.latest_qpos = self.transform_glove_to_v6(glove20)

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
        description='V6 hand rule-based retargeting with real-time Modbus output.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: sim + tuner + RTU Modbus on /dev/ttyUSB0 @ 2 Mbaud, slave 1
  python v6_rule_based_retargeting_realtime.py

  # TCP Modbus
  python v6_rule_based_retargeting_realtime.py --modbus-method tcp \\
      --modbus-host 192.168.1.10 --modbus-tcp-port 502

  # Different serial port and 50 Hz default send rate
  python v6_rule_based_retargeting_realtime.py --modbus-port /dev/ttyACM0 \\
      --modbus-hz 50

  # Sim/tune only, no Modbus
  python v6_rule_based_retargeting_realtime.py --no-modbus
        """
    )
    parser.add_argument('--hand', type=str, default='v6_right')
    parser.add_argument('--side', type=str, choices=['left', 'right'], default='right')
    parser.add_argument('--no-kinematic', action='store_false', dest='kinematic',
                        help='Physics-mode PD drive instead of kinematic snap.')
    parser.add_argument('--no-ergo-viz', action='store_true',
                        help='Disable the ergo / joint-fraction Open3D window.')
    parser.add_argument('--no-tune', action='store_false', dest='tune',
                        help='Disable the tkinter slider window.')
    parser.add_argument('--no-sim', action='store_false', dest='sim',
                        help='Skip Sapien viewer (Modbus + tuner only).')

    # Modbus options
    parser.add_argument('--no-modbus', action='store_false', dest='modbus_enabled',
                        help='Disable Modbus output entirely.')
    parser.add_argument('--modbus-method', choices=['rtu', 'tcp'], default='rtu')
    parser.add_argument('--modbus-port', type=str, default='/dev/ttyUSB0')
    parser.add_argument('--modbus-baud', type=int, default=2_000_000,
                        help='RTU baud rate (default 2 Mbaud, matches the demo).')
    parser.add_argument('--modbus-host', type=str, default='localhost')
    parser.add_argument('--modbus-tcp-port', type=int, default=502)
    parser.add_argument('--modbus-slave-id', type=int, default=1)
    parser.add_argument('--modbus-start-reg', type=int,
                        default=rtu.V6ModbusClient.ADDRESS_CMD_POSITION,
                        help='Position command holding-register address '
                             '(default 0x10 = address_cmd_position).')
    parser.add_argument('--modbus-hz', type=float, default=100.0,
                        help='Default send rate (Hz). Adjustable from the slider window.')

    args = parser.parse_args()

    # --------------------------------------------------------------------
    # Modbus init (soft-fail: keeps running without it)
    # --------------------------------------------------------------------
    modbus = None
    if args.modbus_enabled:
        if not rtu.PYMODBUS_AVAILABLE:
            print('[main] pymodbus not installed — Modbus output disabled. '
                  '(pip install pymodbus)')
        else:
            try:
                modbus = rtu.ModbusBroadcaster(
                    method=args.modbus_method,
                    port=args.modbus_port,
                    baudrate=args.modbus_baud,
                    host=args.modbus_host,
                    tcp_port=args.modbus_tcp_port,
                    slave_id=args.modbus_slave_id,
                    start_register=args.modbus_start_reg,
                    default_hz=args.modbus_hz,
                )
                if modbus.connect():
                    print(f'[main] Modbus connected '
                          f'({args.modbus_method}, slave {args.modbus_slave_id}, '
                          f'start reg {args.modbus_start_reg}).')
                else:
                    print('[main] Modbus connect failed — broadcast disabled. '
                          'Sliders will still run; check wiring and re-Start.')
            except Exception as e:
                print(f'[main] Modbus init error: {e} — broadcast disabled.')
                modbus = None

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
        ru.colorize_articulation(model.hand, _v6_link_classifier)
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
            V6_LOWER_LIMITS, V6_UPPER_LIMITS,
            fingers=['thumb', 'index', 'middle', 'ring', 'pinky'],
            finger_rgb=V6_FINGER_RGB,
            window_title='Manus ergonomics (front) / V6 joint fraction (back)',
        )

    # --------------------------------------------------------------------
    # ROS node + background spin
    # --------------------------------------------------------------------
    rclpy.init()
    node = ManusV6RealtimeNode(side=args.side, modbus=modbus)
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
                scale_layout=V6_SCALE_LAYOUT,
                offset_layout=V6_OFFSET_LAYOUT,
                reversed_scale_keys=V6_REVERSED_SCALE_KEYS,
                scale_magnitude_max=3.5, scale_res=0.1,
                offset_min=-90.0, offset_max=90.0, offset_res=5.0,
                slider_length=200,
                with_load=True,
                with_record=True, default_record_hz=50,
                with_broadcast=True, default_broadcast_hz=args.modbus_hz,
                title='V6 Retargeting Tuning (Realtime Modbus)',
            )
            print('[main] tuning slider window open.')
        except RuntimeError as e:
            print(f'[main] failed to open tuner: {e}')

    mode_str = 'kinematic snap' if args.kinematic else 'physics / PD'
    mb_str = 'modbus ENABLED' if modbus is not None and modbus.connected else 'modbus OFF'
    sim_str = 'sim ON' if model is not None else 'sim OFF'
    print(f"[main] running ({sim_str}, {mode_str}, {mb_str}). "
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
                ergo_viz.update(node.latest_glove20, node.latest_qpos)
                ergo_viz.render()

            if tuner is not None:
                tuner.poll()
    except KeyboardInterrupt:
        pass
    finally:
        if modbus is not None:
            if modbus.broadcasting:
                modbus.stop()
            modbus.disconnect()
        if tuner is not None:
            tuner.close()
        if ergo_viz is not None:
            ergo_viz.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
