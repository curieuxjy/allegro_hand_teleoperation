#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
Manus Glove to V6 Hand Rule-Based Retargeting — REAL-TIME Modbus Output

Like v6_rule_based_retargeting.py but the txt-recording path is replaced
with live Modbus transmission: every glove callback packages the latest
20-D qpos as 16-bit encoder counts and writes them to consecutive holding
registers via pymodbus. Sim viewer, tuning sliders, and ergonomics
visualisation are kept so the operator can verify what is being sent.

CLI (additions over the sim-only script)
  --no-modbus           opt out of Modbus output (sim/tune only)
  --modbus-method       'rtu' (serial) or 'tcp'   (default: rtu)
  --modbus-port         serial device path        (default: /dev/ttyUSB0)
  --modbus-baud         serial baud rate          (default: 115200)
  --modbus-host         TCP host                  (default: localhost)
  --modbus-tcp-port     TCP port                  (default: 502)
  --modbus-slave-id     Modbus slave id           (default: 1)
  --modbus-start-reg    starting holding register (default: 0)
  --modbus-hz           default send rate Hz      (default: 100)
"""

import sys
import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
import sapien.core as sapien
from manus_ros2_msgs.msg import ManusGlove

# Soft import: pymodbus is optional, sim-only mode still works without it.
# Protocol bits (framer, addresses, conversion) match v6_modbus_rtu_demo.py.
try:
    from pymodbus import FramerType
    from pymodbus.client import ModbusSerialClient, ModbusTcpClient
    _PYMODBUS_AVAILABLE = True
except ImportError:
    FramerType = None
    ModbusSerialClient = None
    ModbusTcpClient = None
    _PYMODBUS_AVAILABLE = False

# Path setup for direct script execution
try:
    from .geort.utils.config_utils import get_config
    from .geort.env.hand_debug import HandKinematicModel, HandViewerEnv
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from geort.utils.config_utils import get_config
    from geort.env.hand_debug import HandKinematicModel, HandViewerEnv


# Joint limits for V6 right hand (radians), URDF/config user order:
#   Thumb (joint_00..03) -> Index (joint_10..13) -> Middle (joint_20..23) ->
#   Ring (joint_30..33) -> Pinky (joint_40..43).
V6_UPPER_LIMITS = np.array([
     1.3962,  2.7925,  1.5708,  1.5708,   # thumb
     1.5708,  1.7453,  1.5708,  1.5708,   # index
     1.3962,  1.7453,  1.5708,  1.5708,   # middle
     0.1745,  1.7453,  1.5708,  1.5708,   # ring
     0.0,     0.1745,  1.5708,  1.5708,   # pinky
])

V6_LOWER_LIMITS = np.array([
    -1.3962,  0.0,    -0.0873, -0.0873,   # thumb
    -0.1745,  0.0,    -0.0873, -0.0873,   # index
    -1.3962,  0.0,    -0.0873, -0.0873,   # middle
    -1.3962,  0.0,    -0.0873, -0.0873,   # ring
    -1.7453, -1.5708, -0.0873, -0.0873,   # pinky
])


# ----------------------------------------------------------------------
# Modbus broadcaster
# ----------------------------------------------------------------------
class ModbusBroadcaster:
    """Sends 20-D V6 qpos to a Modbus device as encoder counts.

    Wire protocol (framer/baud/addresses/API) follows v6_modbus_rtu_demo.py:
      * RTU framer (FramerType.RTU), 2_000_000 baud default, 0.1 s timeout
      * `device_id=` parameter for pymodbus 3.7+; `no_response_expected=False`
      * Position command at holding register `address_cmd_position = 0x10`
      * Servo enable via write_coils to `address_servo = 0x03`
      * Signed→unsigned packing via `np.int16 → np.uint16`

    Conversion mirrors the txt-recording path:
        save_frame = qpos.copy()
        save_frame[0]  *= -1     # thumb joint_00 sign-flip
        save_frame[16] *= -1     # pinky joint_40 sign-flip
        counts = round(rad2deg(save_frame) * 4096 / 360)
        regs   = np.array(counts, dtype=np.int16).astype(np.uint16).tolist()
        client.write_registers(0x10, regs, device_id=slave_id,
                               no_response_expected=False)
    """

    DEG_TO_COUNTS = 4096.0 / 360.0

    # Register / coil addresses (match v6_modbus_rtu_demo.py).
    ADDRESS_CMD_POSITION = 0x10
    ADDRESS_CMD_CURRENT  = 0x24
    ADDRESS_SERVO        = 0x03

    # Current limit per finger joint [j0, j1, j2, j3]; same values as the
    # demo's g_cur_set_value. Without this, motors are at 0 torque limit and
    # won't move even though position commands and servo coil are both set.
    DEFAULT_CURRENT_PER_FINGER = [180, 180, 130, 100]

    def __init__(self,
                 method: str = 'rtu',
                 port: str = '/dev/ttyUSB0', baudrate: int = 2_000_000,
                 host: str = 'localhost', tcp_port: int = 502,
                 slave_id: int = 1,
                 start_register: int = None,
                 default_hz: float = 100.0):
        if not _PYMODBUS_AVAILABLE:
            raise RuntimeError(
                "pymodbus is required for Modbus output (pip install pymodbus)"
            )
        self.method = method.lower()
        self.port = port
        self.baudrate = baudrate
        self.host = host
        self.tcp_port = tcp_port
        self.slave_id = slave_id
        # Default to the demo's position-command address; CLI can override.
        self.start_register = (
            start_register if start_register is not None else self.ADDRESS_CMD_POSITION
        )

        if self.method == 'tcp':
            self.client = ModbusTcpClient(host=host, port=tcp_port)
        else:
            self.client = ModbusSerialClient(
                framer=FramerType.RTU,
                port=port,
                baudrate=baudrate,
                parity='N',
                stopbits=1,
                bytesize=8,
                timeout=0.1,
            )

        # Runtime state (mutable from the slider thread)
        self.connected = False
        self.broadcasting = False
        self._bc_t0 = None
        self._bc_target_hz = float(default_hz)
        self._bc_period = 1.0 / self._bc_target_hz
        self._bc_next_t = None
        self.n_sent = 0
        self.send_errors = 0
        self.last_stats = None

    def connect(self):
        """Open the port AND verify a slave responds, matching the demo's
        get_id() flow: try the configured slave_id first, then 1..7 until
        one answers a read of holding register 0.
        """
        # Step 1: open the port / TCP socket.
        try:
            opened = bool(self.client.connect())
        except Exception as e:
            print(f"[modbus] {self.method.upper()} connect raised: {e}")
            self.connected = False
            return False
        if not opened:
            print(f"[modbus] could not open {self.port if self.method == 'rtu' else self.host}")
            if self.method == 'rtu':
                print("[modbus]   - is the cable plugged in?  ls /dev/ttyUSB* /dev/ttyACM*")
                print("[modbus]   - permission denied?  sudo usermod -a -G dialout $USER  "
                      "(re-login required)")
                print(f"[modbus]   - baud is {self.baudrate}; demo uses 2_000_000")
            self.connected = False
            return False

        # Step 2: probe slave IDs. Try the user-specified one first.
        candidates = [self.slave_id] + [i for i in range(1, 8) if i != self.slave_id]
        for sid in candidates:
            try:
                result = self.client.read_holding_registers(
                    address=0, count=1, device_id=sid,
                )
            except Exception as e:
                print(f"[modbus]   probe slave={sid}: exception {e}")
                continue
            if result is None:
                continue
            if hasattr(result, 'isError') and result.isError():
                continue
            # Successful read → this slave is alive.
            if sid != self.slave_id:
                print(f"[modbus] slave {self.slave_id} did not answer; "
                      f"switching to slave {sid}")
            self.slave_id = sid
            self.connected = True
            print(f"[modbus] connected on {self.port if self.method == 'rtu' else self.host}, "
                  f"slave_id={sid}")
            return True

        # No slave responded → port is open but the device isn't talking.
        print(f"[modbus] port opened but no slave responded (tried IDs {candidates})")
        print("[modbus]   - check the slave ID dial on the device")
        print(f"[modbus]   - baud / parity must match firmware "
              f"(we send {self.baudrate} 8N1)")
        print("[modbus]   - the v6 demo expects 2_000_000 baud; "
              "use --modbus-baud to change")
        self.connected = False
        try:
            self.client.close()
        except Exception:
            pass
        return False

    def disconnect(self):
        # Try to leave the device in servo-off for safety before closing.
        try:
            if self.connected:
                self.set_servo(False)
        except Exception:
            pass
        try:
            self.client.close()
        except Exception:
            pass
        self.connected = False

    def set_servo(self, on: bool) -> bool:
        """Toggle the device's servo coil at ADDRESS_SERVO (0x03)."""
        if not self.connected:
            return False
        try:
            result = self.client.write_coils(
                address=self.ADDRESS_SERVO,
                values=[bool(on)],
                device_id=self.slave_id,
            )
            return not (result is None or
                        (hasattr(result, 'isError') and result.isError()))
        except Exception as e:
            print(f"[modbus] set_servo({on}) failed: {e}")
            return False

    def set_current(self, per_finger=None) -> bool:
        """Write 20 current-limit values to ADDRESS_CMD_CURRENT (0x24).
        Without this, motor torque is 0 — servo ON alone is not enough.
        `per_finger` is a 4-element list applied to all 5 fingers."""
        if not self.connected:
            return False
        if per_finger is None:
            per_finger = self.DEFAULT_CURRENT_PER_FINGER
        values = list(per_finger) * 5    # 4 × 5 = 20 (matches demo's g_cur_set_value * FINGER_CNT)
        try:
            result = self.client.write_registers(
                address=self.ADDRESS_CMD_CURRENT,
                values=values,
                device_id=self.slave_id,
                no_response_expected=False,
            )
            return not (result is None or
                        (hasattr(result, 'isError') and result.isError()))
        except Exception as e:
            print(f"[modbus] set_current failed: {e}")
            return False

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
        # Demo workflow before sending position commands:
        #   1. Set current limits (motors need a non-zero torque ceiling)
        #   2. Servo ON (enable the position controller)
        # Both writes are logged so the user can see if either silently failed.
        if self.connected:
            ok_curr = self.set_current()
            ok_servo = self.set_servo(True)
            print(f"[modbus] broadcast start: "
                  f"current={'OK' if ok_curr else 'FAIL'} "
                  f"({self.DEFAULT_CURRENT_PER_FINGER}×5), "
                  f"servo={'ON' if ok_servo else 'FAIL'}")
        self.broadcasting = True

    def stop(self):
        self.broadcasting = False
        elapsed = (time.monotonic() - self._bc_t0) if self._bc_t0 else 0.0
        fps = (self.n_sent / elapsed) if elapsed > 0 else 0.0
        self.last_stats = {
            'n': self.n_sent, 'elapsed': elapsed, 'fps': fps,
            'errors': self.send_errors,
        }
        return self.last_stats

    def send_qpos(self, qpos_rad):
        """Convert qpos and write to position holding registers. Returns
        success bool."""
        if not self.connected:
            self.send_errors += 1
            return False
        save_frame = qpos_rad.copy()
        save_frame[0]  *= -1
        save_frame[16] *= -1
        counts_int16 = np.round(
            np.rad2deg(save_frame) * self.DEG_TO_COUNTS
        ).astype(np.int16)
        # Signed → unsigned 16-bit (two's complement), matches the demo's
        # `np.array(data, dtype=np.int16).astype(np.uint16).tolist()` idiom.
        regs = counts_int16.astype(np.uint16).tolist()
        try:
            result = self.client.write_registers(
                address=self.start_register,
                values=regs,
                device_id=self.slave_id,
                no_response_expected=False,
            )
            if result is None or (hasattr(result, 'isError') and result.isError()):
                self.send_errors += 1
                return False
            self.n_sent += 1
            return True
        except Exception:
            self.send_errors += 1
            return False

    def maybe_send(self, qpos_rad):
        """Drift-corrected scheduler — call every glove callback. Sends only
        when the next scheduled time has been reached."""
        if not self.broadcasting:
            return
        now = time.monotonic()
        if self._bc_next_t is None:
            self.send_qpos(qpos_rad)
            self._bc_next_t = now + self._bc_period
        elif now >= self._bc_next_t:
            self.send_qpos(qpos_rad)
            self._bc_next_t += self._bc_period
            if now > self._bc_next_t:
                self._bc_next_t = now + self._bc_period


# ----------------------------------------------------------------------
# ROS node — identical retargeting math to v6_rule_based_retargeting.py,
# recording replaced by a `modbus` reference whose `maybe_send` is called
# from the glove callback.
# ----------------------------------------------------------------------
class ManusV6RealtimeNode(Node):
    DEFAULT_SCALES = {
        't00_curl': 1.0, 't00_spread': 1.0, 't01': 1.0, 't02': 1.0, 't03': 1.0,
        'i10': 1.0, 'i11': 1.0, 'i12': 1.0, 'i13': 1.0,
        'm20': 1.0, 'm21': 1.0, 'm22': 1.0, 'm23': 1.0,
        'r30': 1.0, 'r31': 1.0, 'r32': 1.0, 'r33': 1.0,
        'p40_curl': 1.0, 'p40_spread': 1.0, 'p41': 1.0, 'p42': 1.0, 'p43': 1.0,
    }
    DEFAULT_OFFSETS = {
        'o_t00': 0.0, 'o_t01': 0.0, 'o_t02': 0.0, 'o_t03': 0.0,
        'o_i10': 0.0, 'o_i11': 0.0, 'o_i12': 0.0, 'o_i13': 0.0,
        'o_m20': 0.0, 'o_m21': 0.0, 'o_m22': 0.0, 'o_m23': 0.0,
        'o_r30': 0.0, 'o_r31': 0.0, 'o_r32': 0.0, 'o_r33': 0.0,
        'o_p40': 0.0, 'o_p41': 0.0, 'o_p42': 0.0, 'o_p43': 0.0,
    }

    TUNING_FILE = Path(__file__).resolve().parent / 'v6_tuning.json'
    RECORDING_DIR = Path(__file__).resolve().parent / 'recordings'

    # Same conversion as ModbusBroadcaster (kept here so recording works
    # even if --no-modbus and the broadcaster instance is None).
    DEG_TO_COUNTS = 4096.0 / 360.0

    T01_MIRROR_BASELINE_DEG = 160.0

    ERGO_MAX_DEG = {'spread': 30.0, 'stretch': 70.0}
    SCALE_NORM_INFO = {
        't00_curl':   (30.0,  'stretch'),
        't00_spread': (60.0,  'spread'),
        't01':        (160.0, 'stretch'),
        't02':        (90.0,  'stretch'),
        't03':        (90.0,  'stretch'),
        'i10':        (90.0,  'spread'),
        'i11':        (100.0, 'stretch'),
        'i12':        (90.0,  'stretch'),
        'i13':        (90.0,  'stretch'),
        'm20':        (60.0,  'spread'),
        'm21':        (100.0, 'stretch'),
        'm22':        (90.0,  'stretch'),
        'm23':        (90.0,  'stretch'),
        'r30':        (80.0,  'spread'),
        'r31':        (100.0, 'stretch'),
        'r32':        (90.0,  'stretch'),
        'r33':        (90.0,  'stretch'),
        'p40_curl':   (80.0,  'stretch'),
        'p40_spread': (20.0,  'spread'),
        'p41':        (90.0,  'stretch'),
        'p42':        (90.0,  'stretch'),
        'p43':        (90.0,  'stretch'),
    }

    _FINGER_OFFSET = {'Thumb': 0, 'Index': 4, 'Middle': 8, 'Ring': 12, 'Pinky': 16}
    _MOTION_IDX = {
        'MCPSpread':  0, 'Spread': 0,
        'MCPStretch': 1, 'PIPStretch': 2, 'DIPStretch': 3,
    }

    def __init__(self, side: str = 'right', modbus: ModbusBroadcaster = None):
        super().__init__('manus_v6_realtime_node')
        self.side = side.lower()
        self.alpha = 0.2
        self.prev_arr = None

        self.scales = dict(self.DEFAULT_SCALES)
        self.offsets = dict(self.DEFAULT_OFFSETS)
        self._try_load_tuning()

        self.latest_qpos = None
        self.latest_glove20 = None
        self._got_first = False

        # Live Modbus output (None if the user passed --no-modbus or pymodbus
        # isn't installed). Sliders flip its `broadcasting` flag on/off.
        self.modbus = modbus

        # Recording state (txt save — independent of Modbus broadcast)
        self.recording = False
        self.recorded_frames = []    # list of np.ndarray (20-D, radians)
        self._rec_t0 = None
        self._rec_target_hz = 120.0
        self._rec_period = 1.0 / 120.0
        self._rec_next_t = None
        self.last_record_stats = None

        topic = f'/manus_glove_{self.side}'
        self.create_subscription(ManusGlove, topic, self.glove_callback, 20)
        self.get_logger().info(f'Subscribed to {topic}')

    # ----- Motion recording (parallel to Modbus broadcast) -------------
    def start_recording(self, target_hz: float = 120.0):
        target_hz = float(max(0.1, min(target_hz, 120.0)))
        self.recorded_frames = []
        self._rec_t0 = time.monotonic()
        self._rec_target_hz = target_hz
        self._rec_period = 1.0 / target_hz
        self._rec_next_t = None
        self.recording = True
        self.get_logger().info(f'Motion recording started @ target {target_hz:.1f} Hz.')

    def stop_recording(self):
        """Stop and write frames to recordings/motion_<timestamp>.txt as
        bracketed encoder counts (same format and sign-flips as the sim
        script: thumb joint_00 and pinky joint_40 are negated only on disk)."""
        self.recording = False
        elapsed = (time.monotonic() - self._rec_t0) if self._rec_t0 else 0.0
        self._rec_t0 = None
        frames = self.recorded_frames
        self.recorded_frames = []
        n = len(frames)
        if n == 0:
            self.get_logger().warn('Recording stopped: no frames captured.')
            return None
        fps = (n / elapsed) if elapsed > 0 else 0.0
        try:
            self.RECORDING_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            path = self.RECORDING_DIR / f'motion_{ts}.txt'
            with open(path, 'w') as f:
                for frame in frames:
                    save_frame = frame.copy()
                    save_frame[0]  *= -1
                    save_frame[16] *= -1
                    counts = np.round(
                        np.rad2deg(save_frame) * self.DEG_TO_COUNTS
                    ).astype(int)
                    f.write('[' + ', '.join(str(c) for c in counts) + ']')
            self.get_logger().info(
                f'Saved {n} frames over {elapsed:.2f}s '
                f'→ {fps:.1f} Hz → {path}'
            )
            self.last_record_stats = {
                'n': n, 'elapsed': elapsed, 'fps': fps, 'path': path,
            }
            return path
        except OSError as e:
            self.get_logger().error(f'Recording save failed: {e}')
            return None

    # ----- Tuning IO ---------------------------------------------------
    def _try_load_tuning(self):
        import json
        if not self.TUNING_FILE.exists():
            return
        try:
            with open(self.TUNING_FILE, 'r') as f:
                data = json.load(f)
            for k, v in (data.get('scales') or {}).items():
                if k in self.scales:
                    self.scales[k] = float(v)
            for k, v in (data.get('offsets') or {}).items():
                if k in self.offsets:
                    self.offsets[k] = float(v)
            self.get_logger().info(f'Loaded tuning from {self.TUNING_FILE}')
        except (OSError, json.JSONDecodeError, ValueError) as e:
            self.get_logger().warn(f'Could not load {self.TUNING_FILE}: {e}')

    def save_tuning(self):
        import json
        try:
            data = {'scales': dict(self.scales), 'offsets': dict(self.offsets)}
            with open(self.TUNING_FILE, 'w') as f:
                json.dump(data, f, indent=2, sort_keys=True)
            self.get_logger().info(f'Saved tuning to {self.TUNING_FILE}')
            return True
        except OSError as e:
            self.get_logger().error(f'Save failed: {e}')
            return False

    def reload_tuning(self):
        """Mutate scales/offsets dicts in place (TuningSliders holds refs)."""
        self.scales.clear()
        self.scales.update(self.DEFAULT_SCALES)
        self.offsets.clear()
        self.offsets.update(self.DEFAULT_OFFSETS)
        self._try_load_tuning()

    # ----- Mapping -----------------------------------------------------
    def _norm(self, key, ergo_val):
        amp, ergo_type = self.SCALE_NORM_INFO[key]
        return self.scales[key] * (ergo_val / self.ERGO_MAX_DEG[ergo_type]) * amp

    def transform_glove_to_v6(self, glove20):
        thumb_vals  = np.array(glove20[0:4],   dtype=float)
        index_vals  = np.array(glove20[4:8],   dtype=float)
        middle_vals = np.array(glove20[8:12],  dtype=float)
        ring_vals   = np.array(glove20[12:16], dtype=float)
        pinky_vals  = np.array(glove20[16:20], dtype=float)

        n = self._norm
        angle_deg = np.array([
            # Thumb
            n('t00_curl', thumb_vals[1]) + n('t00_spread', thumb_vals[0]),
            self.T01_MIRROR_BASELINE_DEG - n('t01', thumb_vals[1]),
            n('t02', thumb_vals[2]),
            n('t03', thumb_vals[3]),
            # Index
            n('i10', index_vals[0]),
            n('i11', index_vals[1]),
            n('i12', index_vals[2]),
            n('i13', index_vals[3]),
            # Middle
            n('m20', middle_vals[0]),
            n('m21', middle_vals[1]),
            n('m22', middle_vals[2]),
            n('m23', middle_vals[3]),
            # Ring (spread negated)
            n('r30', ring_vals[0]),
            n('r31', ring_vals[1]),
            n('r32', ring_vals[2]),
            n('r33', ring_vals[3]),
            # Pinky
            -(n('p40_curl', pinky_vals[1]) - n('p40_spread', pinky_vals[0])),
            -n('p41', pinky_vals[1]),
            n('p42', pinky_vals[2]),
            n('p43', pinky_vals[3]),
        ], dtype=float)

        o = self.offsets
        offset_deg = np.array([
            o['o_t00'], o['o_t01'], o['o_t02'], o['o_t03'],
            o['o_i10'], o['o_i11'], o['o_i12'], o['o_i13'],
            o['o_m20'], o['o_m21'], o['o_m22'], o['o_m23'],
            o['o_r30'], o['o_r31'], o['o_r32'], o['o_r33'],
            o['o_p40'], o['o_p41'], o['o_p42'], o['o_p43'],
        ], dtype=float)
        angle_deg = angle_deg + offset_deg

        arr = np.deg2rad(angle_deg)
        arr = np.clip(arr, V6_LOWER_LIMITS, V6_UPPER_LIMITS)

        if self.prev_arr is None:
            smoothed = arr.copy()
        else:
            smoothed = self.alpha * arr + (1.0 - self.alpha) * self.prev_arr

        # Anti-collision: pinky joint_41 ≤ ring joint_30
        if smoothed[17] > smoothed[12]:
            smoothed[17] = smoothed[12]
        # Middle abduction follows half of index abduction
        # print(smoothed[4], smoothed[8])
        smoothed[4]= smoothed[8] * 1.2


        self.prev_arr = smoothed
        return smoothed

    def _ergonomics_to_array(self, ergonomics_list):
        glove20 = np.zeros(20, dtype=float)
        for ergo in ergonomics_list:
            t = ergo.type
            for finger, offset in self._FINGER_OFFSET.items():
                if t.startswith(finger):
                    motion = t[len(finger):]
                    if motion in self._MOTION_IDX:
                        glove20[offset + self._MOTION_IDX[motion]] = ergo.value
                    break
        return glove20

    def glove_callback(self, msg: ManusGlove):
        if len(msg.ergonomics) == 0:
            return
        glove20 = self._ergonomics_to_array(msg.ergonomics)
        self.latest_glove20 = glove20
        self.latest_qpos = self.transform_glove_to_v6(glove20)

        # Real-time Modbus dispatch (rate-limited by the broadcaster).
        if self.modbus is not None:
            self.modbus.maybe_send(self.latest_qpos)

        # Recording (drift-corrected scheduler — independent of Modbus).
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


# ----------------------------------------------------------------------
# Sapien helpers / colors / ergo viz — identical to v6 sim script
# ----------------------------------------------------------------------
def _setup_sapien_scene():
    engine = sapien.Engine()
    renderer = sapien.VulkanRenderer()
    engine.set_renderer(renderer)

    scene_cfg = sapien.SceneConfig()
    scene_cfg.default_dynamic_friction = 1.0
    scene_cfg.default_static_friction = 1.0
    scene_cfg.default_restitution = 0.00
    scene_cfg.contact_offset = 0.02
    scene_cfg.enable_pcm = False
    scene_cfg.solver_iterations = 50
    scene_cfg.solver_velocity_iterations = 2
    scene = engine.create_scene(scene_cfg)
    return engine, renderer, scene


def _tune_v6_drives(model: HandKinematicModel,
                    kp: float = 400.0, kd: float = 30.0,
                    force_limit: float = 10.0):
    for j in model.all_joints:
        j.set_drive_property(kp, kd, force_limit=force_limit)


FINGER_COLORS = {
    'thumb':  [0.95, 0.30, 0.30, 1.0],
    'index':  [0.30, 0.55, 0.95, 1.0],
    'middle': [0.30, 0.85, 0.40, 1.0],
    'ring':   [0.95, 0.65, 0.20, 1.0],
    'pinky':  [0.75, 0.40, 0.85, 1.0],
}


def _set_link_color(link, rgba):
    try:
        bodies = link.get_visual_bodies()
    except AttributeError:
        return
    for body in bodies:
        try:
            for shape in body.get_render_shapes():
                shape.material.set_base_color(rgba)
            continue
        except (AttributeError, TypeError):
            pass
        try:
            body.set_color(rgba[:3])
        except (AttributeError, TypeError):
            pass


def _colorize_v6_fingers(articulation):
    for link in articulation.get_links():
        name = link.get_name()
        for finger, rgba in FINGER_COLORS.items():
            if name.startswith(finger + '_'):
                _set_link_color(link, rgba)
                break


class ErgonomicsBarViz:
    """Open3D dual-row bar chart (front = glove deg, back = joint fraction)."""

    FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky']
    MOTIONS = ['spread', 'mcp', 'pip', 'dip']

    def __init__(self, joint_lower, joint_upper,
                 deg_scale: float = 0.01, frac_scale: float = 1.2,
                 bar_width: float = 0.35, bar_depth: float = 0.35,
                 init_height: float = 1.0,
                 finger_spacing: float = 2.6, motion_spacing: float = 0.5,
                 row_z_gap: float = 1.6):
        self.joint_lower = np.asarray(joint_lower, dtype=float)
        self.joint_upper = np.asarray(joint_upper, dtype=float)
        self.joint_range = np.where(
            (self.joint_upper - self.joint_lower) > 1e-9,
            self.joint_upper - self.joint_lower, 1.0,
        )
        self.deg_scale = deg_scale
        self.frac_scale = frac_scale
        self.init_height = init_height

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name='Manus ergonomics (front) / V6 joint fraction (back)',
            width=1200, height=620,
        )

        self.boxes_top, self.init_verts_top = [], []
        self.boxes_bot, self.init_verts_bot = [], []
        self.bar_x = np.zeros(20)
        z_top = -row_z_gap / 2
        z_bot = +row_z_gap / 2

        for i in range(20):
            f_idx, m_idx = i // 4, i % 4
            finger = self.FINGERS[f_idx]
            x = f_idx * finger_spacing + m_idx * motion_spacing
            self.bar_x[i] = x
            base = FINGER_COLORS[finger][:3]
            brightness = 1.0 - 0.18 * m_idx
            color = [c * brightness for c in base]
            for boxes, init_verts, z_center in (
                (self.boxes_top, self.init_verts_top, z_top),
                (self.boxes_bot, self.init_verts_bot, z_bot),
            ):
                box = o3d.geometry.TriangleMesh.create_box(
                    width=bar_width, height=init_height, depth=bar_depth,
                )
                box.translate([x - bar_width / 2, 0.0, z_center - bar_depth / 2])
                box.compute_vertex_normals()
                box.paint_uniform_color(color)
                boxes.append(box)
                init_verts.append(np.asarray(box.vertices).copy())
                self.vis.add_geometry(box)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(frame)
        x_min = -0.5
        x_max = 4 * finger_spacing + 3 * motion_spacing + 0.5
        ref = o3d.geometry.LineSet()
        ref.points = o3d.utility.Vector3dVector([
            [x_min, 0.0, z_top], [x_max, 0.0, z_top],
            [x_min, 0.0, z_bot], [x_max, 0.0, z_bot],
            [x_min, frac_scale, z_bot], [x_max, frac_scale, z_bot],
        ])
        ref.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3], [4, 5]])
        ref.colors = o3d.utility.Vector3dVector([
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.85, 0.25, 0.25],
        ])
        self.vis.add_geometry(ref)

        self.update(np.zeros(20), np.zeros(20))
        self._first_render = True

    def _resize_bar(self, box, init_verts, signed_height):
        if abs(signed_height) < 1e-3:
            signed_height = 1e-3 if signed_height >= 0 else -1e-3
        scale = signed_height / self.init_height
        verts = init_verts.copy()
        verts[:, 1] *= scale
        box.vertices = o3d.utility.Vector3dVector(verts)
        box.compute_vertex_normals()
        self.vis.update_geometry(box)

    def update(self, glove20, qpos20):
        if glove20 is not None and len(glove20) >= 20:
            for i in range(20):
                h = float(glove20[i]) * self.deg_scale
                self._resize_bar(self.boxes_top[i], self.init_verts_top[i], h)
        if qpos20 is not None and len(qpos20) >= 20:
            q = np.asarray(qpos20, dtype=float)
            normalized = (q - self.joint_lower) / self.joint_range
            for i in range(20):
                h = float(normalized[i]) * self.frac_scale
                self._resize_bar(self.boxes_bot[i], self.init_verts_bot[i], h)

    def render(self):
        if self._first_render:
            ctr = self.vis.get_view_control()
            ctr.set_lookat([7.0, 0.8, 0.0])
            ctr.set_front([0.0, -0.3, 1.0])
            ctr.set_up([0.0, 1.0, 0.0])
            ctr.set_zoom(0.5)
            self._first_render = False
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Tkinter slider window — same scale/offset layout as the sim script, but
# the bottom row exposes a Modbus broadcast toggle + send-rate spinbox.
# ----------------------------------------------------------------------
class TuningSliders:
    _SCALE_LAYOUT = [
        ('Thumb', [
            ('t00_curl',   '00 ← curl'),
            ('t00_spread', '00 ← spread'),
            ('t01',        '01 (curl)'),
            ('t02',        '02 (MCP)'),
            ('t03',        '03 (IP)'),
        ]),
        ('Index', [('i10', '10 (spread)'), ('i11', '11 (MCP)'),
                   ('i12', '12 (PIP)'),    ('i13', '13 (DIP)')]),
        ('Middle', [('m20', '20 (spread)'), ('m21', '21 (MCP)'),
                    ('m22', '22 (PIP)'),    ('m23', '23 (DIP)')]),
        ('Ring',  [('r30', '30 (spread)'), ('r31', '31 (MCP)'),
                   ('r32', '32 (PIP)'),    ('r33', '33 (DIP)')]),
        ('Pinky', [
            ('p40_curl',   '40 ← curl'),
            ('p40_spread', '40 ← spread'),
            ('p41',        '41 (MCP)'),
            ('p42',        '42 (PIP)'),
            ('p43',        '43 (DIP)'),
        ]),
    ]
    _OFFSET_LAYOUT = [
        ('Thumb',  ['o_t00', 'o_t01', 'o_t02', 'o_t03']),
        ('Index',  ['o_i10', 'o_i11', 'o_i12', 'o_i13']),
        ('Middle', ['o_m20', 'o_m21', 'o_m22', 'o_m23']),
        ('Ring',   ['o_r30', 'o_r31', 'o_r32', 'o_r33']),
        ('Pinky',  ['o_p40', 'o_p41', 'o_p42', 'o_p43']),
    ]

    _FONT_HEADER  = ('TkDefaultFont', 13, 'bold')
    _FONT_SECTION = ('TkDefaultFont', 11, 'italic')
    _FONT_LABEL   = ('TkDefaultFont', 11)
    _FONT_SLIDER  = ('TkDefaultFont', 10)
    _FONT_BUTTON  = ('TkDefaultFont', 11, 'bold')

    _REVERSED_SCALE_KEYS = {'p40_curl'}

    def __init__(self, node,
                 scale_magnitude_max=3.5, scale_res=0.1,
                 offset_min=-90.0, offset_max=90.0, offset_res=5.0,
                 slider_length=200,
                 default_modbus_hz=100):
        try:
            import tkinter as tk
        except ImportError as e:
            raise RuntimeError(
                "tkinter is required for --tune (try: sudo apt install python3-tk)"
            ) from e
        self.tk = tk
        self.node = node
        self.scale_magnitude_max = scale_magnitude_max
        self.scale_res = scale_res
        self.offset_min = offset_min
        self.offset_max = offset_max
        self.offset_res = offset_res
        self.slider_length = slider_length

        self.root = tk.Tk()
        self.root.title('V6 Retargeting Tuning (Realtime Modbus)')
        self._slider_vars = {}

        outer = tk.Frame(self.root)
        outer.pack(padx=8, pady=8)

        for col_idx, (finger, scale_pairs) in enumerate(self._SCALE_LAYOUT):
            col = tk.LabelFrame(outer, text=finger,
                                font=self._FONT_HEADER, padx=6, pady=6)
            col.grid(row=0, column=col_idx, padx=4, sticky='n')
            tk.Label(col, text='Step 2 scales  (sign locked)',
                     font=self._FONT_SECTION, fg='#555').pack(anchor='w')
            for key, label in scale_pairs:
                self._make_scale_slider(col, key, label)
            tk.Frame(col, height=2, bg='#bbb').pack(fill='x', pady=6)
            tk.Label(col, text='Step 3 offsets (deg, signed)',
                     font=self._FONT_SECTION, fg='#555').pack(anchor='w')
            _, offset_keys = self._OFFSET_LAYOUT[col_idx]
            for okey in offset_keys:
                self._make_offset_slider(col, okey, okey[2:])

        # Row 1: Save / Load / Reset
        btn_row = tk.Frame(self.root)
        btn_row.pack(pady=(0, 4))
        tk.Button(btn_row, text='Save', font=self._FONT_BUTTON,
                  width=10, command=self._on_save).pack(side='left', padx=4)
        tk.Button(btn_row, text='Load', font=self._FONT_BUTTON,
                  width=10, command=self._on_load).pack(side='left', padx=4)
        tk.Button(btn_row, text='Reset to defaults', font=self._FONT_BUTTON,
                  width=18, command=self._reset_to_defaults).pack(side='left', padx=4)

        # Row 2: Modbus broadcast toggle + Hz
        bc_row = tk.Frame(self.root)
        bc_row.pack(pady=(0, 4))
        self._bc_button = tk.Button(
            bc_row, text='● Start Broadcast', font=self._FONT_BUTTON,
            width=20, fg='#a00', command=self._on_broadcast_toggle,
        )
        self._bc_button.pack(side='left', padx=4)
        tk.Label(bc_row, text='Hz:', font=self._FONT_LABEL).pack(side='left', padx=(10, 2))
        self._bc_hz_var = tk.IntVar(value=int(default_modbus_hz))
        self._bc_hz_spin = tk.Spinbox(
            bc_row, from_=1, to=120, increment=1,
            textvariable=self._bc_hz_var, width=5,
            font=self._FONT_LABEL,
        )
        self._bc_hz_spin.pack(side='left', padx=2)
        self._bc_status_var = tk.StringVar(value='')
        tk.Label(bc_row, textvariable=self._bc_status_var,
                 font=self._FONT_LABEL, fg='#a00').pack(side='left', padx=4)

        # Row 3: Motion recording (txt save) — independent of Modbus
        rec_row = tk.Frame(self.root)
        rec_row.pack(pady=(0, 4))
        self._rec_button = tk.Button(
            rec_row, text='● Start Record', font=self._FONT_BUTTON,
            width=20, fg='#a00', command=self._on_record_toggle,
        )
        self._rec_button.pack(side='left', padx=4)
        tk.Label(rec_row, text='Hz:', font=self._FONT_LABEL).pack(side='left', padx=(10, 2))
        self._rec_hz_var = tk.IntVar(value=50)
        self._rec_hz_spin = tk.Spinbox(
            rec_row, from_=1, to=120, increment=1,
            textvariable=self._rec_hz_var, width=5,
            font=self._FONT_LABEL,
        )
        self._rec_hz_spin.pack(side='left', padx=2)
        self._rec_status_var = tk.StringVar(value='')
        tk.Label(rec_row, textvariable=self._rec_status_var,
                 font=self._FONT_LABEL, fg='#a00').pack(side='left', padx=4)

        # General Save/Reset status
        self._status_var = tk.StringVar(value='')
        tk.Label(self.root, textvariable=self._status_var,
                 font=self._FONT_LABEL, fg='#0a0').pack(pady=(0, 4))

        self._tick_bc_status()
        self._tick_rec_status()

    # ----- slider builders ---------------------------------------------
    def _make_scale_slider(self, parent, key, label):
        factory_default = self.node.DEFAULT_SCALES[key]
        current = self.node.scales[key]
        mag = self.scale_magnitude_max
        if factory_default < 0:
            mn, mx = -mag, 0.0
            if current > 0:
                self.node.scales[key] = -current
        else:
            mn, mx = 0.0, mag
            if current < 0:
                self.node.scales[key] = -current
        reverse = key in self._REVERSED_SCALE_KEYS
        self._make_slider(parent, self.node.scales, key, label,
                          mn, mx, self.scale_res, reverse=reverse)

    def _make_offset_slider(self, parent, key, label):
        self._make_slider(parent, self.node.offsets, key, label,
                          self.offset_min, self.offset_max, self.offset_res)

    def _make_slider(self, parent, store, key, label, mn, mx, res, reverse=False):
        tk = self.tk
        row = tk.Frame(parent)
        row.pack(fill='x', pady=2)
        tk.Label(row, text=label, width=12, anchor='w',
                 font=self._FONT_LABEL).pack(side='left')
        var = tk.DoubleVar(value=store[key])

        def on_change(value, k=key, s=store):
            try:
                s[k] = float(value)
            except (TypeError, ValueError):
                pass

        from_, to_ = (mx, mn) if reverse else (mn, mx)
        slider = tk.Scale(row, from_=from_, to=to_, resolution=res,
                          orient='horizontal', variable=var,
                          length=self.slider_length,
                          showvalue=True, command=on_change,
                          font=self._FONT_SLIDER)
        slider.pack(side='left')
        self._slider_vars[(id(store), key)] = (var, store)

    # ----- buttons -----------------------------------------------------
    def _reset_to_defaults(self):
        for k, v in self.node.DEFAULT_SCALES.items():
            self.node.scales[k] = v
            sv = self._slider_vars.get((id(self.node.scales), k))
            if sv is not None:
                sv[0].set(v)
        for k, v in self.node.DEFAULT_OFFSETS.items():
            self.node.offsets[k] = v
            sv = self._slider_vars.get((id(self.node.offsets), k))
            if sv is not None:
                sv[0].set(v)
        self._flash_status('Reset to defaults.')

    def _on_save(self):
        ok = self.node.save_tuning()
        if ok:
            self._flash_status(f'Saved → {self.node.TUNING_FILE.name}')
        else:
            self._flash_status('Save failed (see console).')

    def _on_load(self):
        self.node.reload_tuning()
        for (store_id, key), (var, store) in self._slider_vars.items():
            if key in store:
                var.set(store[key])
        self._flash_status(f'Loaded ← {self.node.TUNING_FILE.name}')

    def _on_broadcast_toggle(self):
        mb = self.node.modbus
        if mb is None:
            self._flash_status('Modbus not available (see startup logs).')
            return
        if mb.broadcasting:
            stats = mb.stop()
            self._bc_button.config(text='● Start Broadcast', fg='#a00')
            err_str = f', {stats["errors"]} err' if stats['errors'] > 0 else ''
            self._flash_status(
                f'Stopped — sent {stats["n"]} @ {stats["fps"]:.1f} Hz{err_str}',
                ms=6000,
            )
        else:
            try:
                hz = float(self._bc_hz_var.get())
            except (ValueError, self.tk.TclError):
                hz = 100.0
            mb.start(target_hz=hz)
            self._bc_button.config(text='■ Stop Broadcast', fg='#080')

    def _tick_bc_status(self):
        mb = self.node.modbus
        if mb is not None and mb.broadcasting:
            err_str = f'  ({mb.send_errors} err)' if mb.send_errors > 0 else ''
            self._bc_status_var.set(
                f'TX {mb.n_sent} frames  (target {mb._bc_target_hz:.0f} Hz){err_str}'
            )
        else:
            self._bc_status_var.set('')
        try:
            self.root.after(200, self._tick_bc_status)
        except self.tk.TclError:
            pass

    def _on_record_toggle(self):
        if self.node.recording:
            path = self.node.stop_recording()
            self._rec_button.config(text='● Start Record', fg='#a00')
            if path is not None:
                stats = self.node.last_record_stats or {}
                n = stats.get('n', 0)
                fps = stats.get('fps', 0.0)
                self._flash_status(
                    f'{n} frames @ {fps:.1f} Hz → {path.name}', ms=6000,
                )
            else:
                self._flash_status('Recording stopped (no frames).')
        else:
            try:
                hz = float(self._rec_hz_var.get())
            except (ValueError, self.tk.TclError):
                hz = 50.0
            self.node.start_recording(target_hz=hz)
            self._rec_button.config(text='■ Stop Record', fg='#080')

    def _tick_rec_status(self):
        if self.node.recording:
            n = len(self.node.recorded_frames)
            target = getattr(self.node, '_rec_target_hz', 0.0)
            self._rec_status_var.set(f'REC {n} frames  (target {target:.0f} Hz)')
        else:
            self._rec_status_var.set('')
        try:
            self.root.after(200, self._tick_rec_status)
        except self.tk.TclError:
            pass

    def _flash_status(self, msg, ms=3000):
        self._status_var.set(msg)
        try:
            self.root.after(ms, lambda: self._status_var.set(''))
        except self.tk.TclError:
            pass

    def poll(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            pass

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def _apply_qpos(model: HandKinematicModel, qpos_user, kinematic: bool):
    if kinematic:
        clipped = np.clip(qpos_user,
                          model.joint_lower_limit + 1e-3,
                          model.joint_upper_limit - 1e-3)
        qpos_sim = model.convert_user_order_to_sim_order(clipped)
        model.hand.set_qpos(qpos_sim)
        model.hand.set_qvel(np.zeros_like(qpos_sim))
    else:
        model.set_qpos_target(qpos_user)


def main():
    parser = argparse.ArgumentParser(
        description='V6 hand rule-based retargeting with real-time Modbus output.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: sim + tuner + RTU Modbus on /dev/ttyUSB0 @ 115200, slave 1
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
                        help='RTU baud rate (default 2 Mbaud, matches v6_modbus_rtu_demo.py).')
    parser.add_argument('--modbus-host', type=str, default='localhost')
    parser.add_argument('--modbus-tcp-port', type=int, default=502)
    parser.add_argument('--modbus-slave-id', type=int, default=1)
    parser.add_argument('--modbus-start-reg', type=int,
                        default=ModbusBroadcaster.ADDRESS_CMD_POSITION,
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
        if not _PYMODBUS_AVAILABLE:
            print('[main] pymodbus not installed — Modbus output disabled. '
                  '(pip install pymodbus)')
        else:
            try:
                modbus = ModbusBroadcaster(
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
        engine, renderer, scene = _setup_sapien_scene()
        config = get_config(args.hand)
        print(f"[main] loading sim config: {config['name']}")
        model = HandKinematicModel.build_from_config(config, scene=scene, render=False)
        model.hand.set_root_pose(sapien.Pose([0.0, 0.0, 0.35], [0.695, 0.0, -0.718, 0.0]))
        _colorize_v6_fingers(model.hand)
        if not args.kinematic:
            _tune_v6_drives(model)
        links = [info['link'] for info in config['fingertip_link']]
        offsets = [info['center_offset'] for info in config['fingertip_link']]
        model.initialize_keypoint(links, offsets)
        viewer_env = HandViewerEnv([model], scene=scene, renderer=renderer)

    # --------------------------------------------------------------------
    # Optional ergo viz
    # --------------------------------------------------------------------
    ergo_viz = None
    if not args.no_ergo_viz:
        ergo_viz = ErgonomicsBarViz(V6_LOWER_LIMITS, V6_UPPER_LIMITS)

    # --------------------------------------------------------------------
    # ROS node + background spin
    # --------------------------------------------------------------------
    rclpy.init()
    node = ManusV6RealtimeNode(side=args.side, modbus=modbus)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # --------------------------------------------------------------------
    # Optional tuner window
    # --------------------------------------------------------------------
    tuner = None
    if args.tune:
        try:
            tuner = TuningSliders(node, default_modbus_hz=args.modbus_hz)
            print('[main] tuning slider window open.')
        except RuntimeError as e:
            print(f'[main] failed to open tuner: {e}')

    mode_str = 'kinematic snap' if args.kinematic else 'physics / PD'
    mb_str = 'modbus ENABLED' if modbus is not None and modbus.connected else 'modbus OFF'
    sim_str = 'sim ON' if model is not None else 'sim OFF'
    print(f"[main] running ({sim_str}, {mode_str}, {mb_str}). "
          f"Waiting for /manus_glove_{args.side} ...")

    # --------------------------------------------------------------------
    # Main loop
    # --------------------------------------------------------------------
    try:
        while True:
            qpos = node.latest_qpos
            if qpos is not None and model is not None:
                _apply_qpos(model, qpos, args.kinematic)

            if viewer_env is not None:
                if args.kinematic:
                    viewer_env._update_tip_positions()
                    viewer_env._update_axis_markers()
                    viewer_env.scene.update_render()
                    viewer_env.viewer.render()
                else:
                    viewer_env.update()
            else:
                # No sim window — sleep a bit so we don't busy-loop.
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
