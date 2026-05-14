#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
Hardware-facing utilities for the V6 hand realtime / replay scripts.

Wraps pymodbus with the V6-specific register layout, current-limit and
servo-enable workflow, and sign-flip conversion. Used by:

  * v6_rule_based_retargeting_realtime.py — live broadcast from glove
  * v6_replay_motion.py                   — playback of a recorded txt

Provided:
  * PYMODBUS_AVAILABLE       — flag, False if pymodbus isn't installed
  * V6ModbusClient           — connect, set_servo, set_current, send_*
  * ModbusBroadcaster        — V6ModbusClient + drift-corrected broadcast
  * MotionReplayer           — V6ModbusClient + threaded playback
  * parse_motion_file(path)  — parses the recorded `[v0,...,v19][...]` text
"""

import re
import threading
import time
from pathlib import Path

import numpy as np

# Soft import: keep working in --no-modbus mode without pymodbus installed.
try:
    from pymodbus import FramerType
    from pymodbus.client import ModbusSerialClient, ModbusTcpClient
    PYMODBUS_AVAILABLE = True
except ImportError:
    FramerType = None
    ModbusSerialClient = None
    ModbusTcpClient = None
    PYMODBUS_AVAILABLE = False


# ----------------------------------------------------------------------
# Shared low-level Modbus client for the V6 hand
# ----------------------------------------------------------------------
class V6ModbusClient:
    """Connect to the V6 Modbus device and send position / current / servo
    commands. Conventions match v6_modbus_rtu_demo.py:

      * RTU framer (FramerType.RTU), 2_000_000 baud default, 0.1 s timeout
      * pymodbus 3.7+ API: `device_id=`, `no_response_expected=False`
      * Position command at holding register `ADDRESS_CMD_POSITION` (0x10)
      * Current limit  at holding register `ADDRESS_CMD_CURRENT`  (0x24)
      * Servo enable   via write_coils to `ADDRESS_SERVO`         (0x03)
      * `send_qpos()`  applies sign flip on thumb_00 / pinky_40 then
                       converts rad → encoder counts → uint16 packing.
    """

    DEG_TO_COUNTS = 4096.0 / 360.0

    ADDRESS_CMD_POSITION = 0x10
    ADDRESS_CMD_CURRENT  = 0x24
    ADDRESS_SERVO        = 0x03

    # Current limit per finger joint [j0, j1, j2, j3]; broadcast to all 5
    # fingers as the demo's g_cur_set_value. Without this, motors are at
    # zero torque limit and ignore position commands even with servo ON.
    DEFAULT_CURRENT_PER_FINGER = [180, 180, 130, 100]

    # Indices in the 20-D qpos that need their sign flipped on the wire
    # (thumb joint_00 and pinky joint_40 — base-swing joints whose URDF
    # convention is opposite the motor's positive direction).
    SIGN_FLIP_INDICES = (0, 16)

    def __init__(self,
                 method: str = 'rtu',
                 port: str = '/dev/ttyUSB0', baudrate: int = 2_000_000,
                 host: str = 'localhost', tcp_port: int = 502,
                 slave_id: int = 1,
                 start_register: int = None):
        if not PYMODBUS_AVAILABLE:
            raise RuntimeError(
                "pymodbus is required for Modbus output "
                "(pip install pymodbus pyserial)"
            )
        self.method = method.lower()
        self.port = port
        self.baudrate = baudrate
        self.host = host
        self.tcp_port = tcp_port
        self.slave_id = slave_id
        self.start_register = (
            start_register if start_register is not None
            else self.ADDRESS_CMD_POSITION
        )

        if self.method == 'tcp':
            self.client = ModbusTcpClient(host=host, port=tcp_port)
        else:
            self.client = ModbusSerialClient(
                framer=FramerType.RTU,
                port=port, baudrate=baudrate,
                parity='N', stopbits=1, bytesize=8,
                timeout=0.1,
            )

        self.connected = False
        self.send_errors = 0

    # ----- connection ------------------------------------------------
    def connect(self) -> bool:
        """Open the port (or TCP socket) and probe for a responsive slave.
        Tries the configured slave_id first, then 1..7 (matches the demo's
        get_id() behavior). Returns True on success."""
        try:
            opened = bool(self.client.connect())
        except Exception as e:
            print(f"[modbus] {self.method.upper()} connect raised: {e}")
            self.connected = False
            return False
        if not opened:
            tag = self.port if self.method == 'rtu' else self.host
            print(f"[modbus] could not open {tag}")
            if self.method == 'rtu':
                print("[modbus]   - is the cable plugged in?  "
                      "ls /dev/ttyUSB* /dev/ttyACM*")
                print("[modbus]   - permission denied?  "
                      "sudo usermod -a -G dialout $USER  (re-login required)")
                print(f"[modbus]   - baud is {self.baudrate}; "
                      "demo uses 2_000_000")
            self.connected = False
            return False

        candidates = [self.slave_id] + [i for i in range(1, 8) if i != self.slave_id]
        for sid in candidates:
            try:
                r = self.client.read_holding_registers(
                    address=0, count=1, device_id=sid,
                )
            except Exception as e:
                print(f"[modbus]   probe slave={sid}: exception {e}")
                continue
            if r is None or (hasattr(r, 'isError') and r.isError()):
                continue
            if sid != self.slave_id:
                print(f"[modbus] slave {self.slave_id} did not answer; "
                      f"switching to slave {sid}")
            self.slave_id = sid
            self.connected = True
            tag = self.port if self.method == 'rtu' else self.host
            print(f"[modbus] connected on {tag}, slave_id={sid}")
            return True

        print(f"[modbus] port opened but no slave responded "
              f"(tried IDs {candidates})")
        print("[modbus]   - check the slave ID dial on the device")
        print(f"[modbus]   - baud / parity must match firmware "
              f"(we send {self.baudrate} 8N1)")
        try:
            self.client.close()
        except Exception:
            pass
        self.connected = False
        return False

    def disconnect(self):
        """Servo-off the device (best effort) then close the port."""
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

    # ----- device ops ------------------------------------------------
    def set_servo(self, on: bool) -> bool:
        if not self.connected:
            return False
        try:
            r = self.client.write_coils(
                address=self.ADDRESS_SERVO,
                values=[bool(on)],
                device_id=self.slave_id,
            )
            return not (r is None or
                        (hasattr(r, 'isError') and r.isError()))
        except Exception as e:
            print(f"[modbus] set_servo({on}) failed: {e}")
            return False

    def set_current(self, per_finger=None) -> bool:
        """Write 20 current-limit values to ADDRESS_CMD_CURRENT.
        Without this the motors have zero torque limit even with servo ON,
        so position commands are silently ignored."""
        if not self.connected:
            return False
        if per_finger is None:
            per_finger = self.DEFAULT_CURRENT_PER_FINGER
        values = list(per_finger) * 5
        try:
            r = self.client.write_registers(
                address=self.ADDRESS_CMD_CURRENT,
                values=values,
                device_id=self.slave_id,
                no_response_expected=False,
            )
            return not (r is None or
                        (hasattr(r, 'isError') and r.isError()))
        except Exception as e:
            print(f"[modbus] set_current failed: {e}")
            return False

    # ----- position commands -----------------------------------------
    def send_counts(self, counts) -> bool:
        """Send 20 signed-16-bit encoder counts (already in motor frame,
        i.e. sign-flips already applied) to the position registers."""
        if not self.connected:
            self.send_errors += 1
            return False
        regs = np.array(counts, dtype=np.int16).astype(np.uint16).tolist()
        try:
            r = self.client.write_registers(
                address=self.start_register,
                values=regs,
                device_id=self.slave_id,
                no_response_expected=False,
            )
            if r is None or (hasattr(r, 'isError') and r.isError()):
                self.send_errors += 1
                return False
            return True
        except Exception:
            self.send_errors += 1
            return False

    def send_qpos(self, qpos_rad) -> bool:
        """Convert a 20-D radian qpos to motor counts (with sign flips) and
        write to the position registers."""
        save_frame = np.asarray(qpos_rad, dtype=float).copy()
        for i in self.SIGN_FLIP_INDICES:
            save_frame[i] *= -1
        counts = np.round(
            np.rad2deg(save_frame) * self.DEG_TO_COUNTS
        ).astype(np.int16)
        return self.send_counts(counts)


# ----------------------------------------------------------------------
# Live-broadcast helper (used by v6_rule_based_retargeting_realtime.py)
# ----------------------------------------------------------------------
class ModbusBroadcaster(V6ModbusClient):
    """V6ModbusClient + a drift-corrected broadcast scheduler.

    `maybe_send(qpos)` is called every glove callback; the scheduler decides
    whether enough time has passed since the last transmission to send the
    next frame at the target Hz. Stats (n_sent, send_errors, fps) are
    surfaced to the tuner UI."""

    def __init__(self, default_hz: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self.broadcasting = False
        self._bc_t0 = None
        self._bc_target_hz = float(default_hz)
        self._bc_period = 1.0 / self._bc_target_hz
        self._bc_next_t = None
        self.n_sent = 0
        self.last_stats = None

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
        # Demo workflow: currents first (torque limit), then servo on.
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
            'n': self.n_sent, 'elapsed': elapsed,
            'fps': fps, 'errors': self.send_errors,
        }
        return self.last_stats

    def maybe_send(self, qpos_rad):
        """Drift-corrected scheduler — call every glove callback. Advances
        `_bc_next_t` by exactly one period so average rate converges to
        target even when the glove tick isn't an even divisor of period."""
        if not self.broadcasting:
            return
        now = time.monotonic()
        if self._bc_next_t is None:
            if self.send_qpos(qpos_rad):
                self.n_sent += 1
            self._bc_next_t = now + self._bc_period
        elif now >= self._bc_next_t:
            if self.send_qpos(qpos_rad):
                self.n_sent += 1
            self._bc_next_t += self._bc_period
            if now > self._bc_next_t:
                self._bc_next_t = now + self._bc_period


# ----------------------------------------------------------------------
# Playback helper (used by v6_replay_motion.py)
# ----------------------------------------------------------------------
class MotionReplayer(V6ModbusClient):
    """V6ModbusClient + a daemon-thread playback scheduler.

    Frames are integer counts (already sign-flipped at record time), so we
    use `send_counts()` directly instead of `send_qpos()`. Supports loop,
    pause / resume via threading.Events, and a callback on completion."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.playing = False
        self.current_idx = 0
        self.total_frames = 0
        self._stop_evt = threading.Event()
        self._pause_evt = threading.Event()
        self._pause_evt.set()           # set = not paused
        self._thread = None

    def play(self, frames, target_hz: float = 50.0,
             loop: bool = False, on_done=None) -> bool:
        if self.playing:
            return False
        if not self.connected:
            print("[replay] not connected — nothing to play to.")
            return False
        self.total_frames = len(frames)
        self.current_idx = 0
        self.send_errors = 0
        self._stop_evt.clear()
        self._pause_evt.set()

        ok_curr = self.set_current()
        ok_servo = self.set_servo(True)
        print(f"[replay] starting: current={'OK' if ok_curr else 'FAIL'}, "
              f"servo={'ON' if ok_servo else 'FAIL'}, "
              f"{self.total_frames} frames @ {target_hz:.1f} Hz, "
              f"loop={loop}")

        self.playing = True
        self._thread = threading.Thread(
            target=self._play_loop,
            args=(frames, target_hz, loop, on_done),
            daemon=True,
        )
        self._thread.start()
        return True

    def _play_loop(self, frames, target_hz, loop, on_done):
        period = 1.0 / max(target_hz, 0.1)
        try:
            while not self._stop_evt.is_set():
                t_start = time.monotonic()
                for i, frame in enumerate(frames):
                    if self._stop_evt.is_set():
                        break
                    self._pause_evt.wait()
                    target_t = t_start + i * period
                    sleep = target_t - time.monotonic()
                    if sleep > 0:
                        time.sleep(sleep)
                    self.send_counts(frame)
                    self.current_idx = i + 1
                if not loop:
                    break
                time.sleep(0.01)
        finally:
            self.playing = False
            print(f"[replay] done — {self.current_idx} frames sent, "
                  f"{self.send_errors} errors.")
            if on_done is not None:
                try:
                    on_done()
                except Exception:
                    pass

    def pause(self):
        self._pause_evt.clear()

    def resume(self):
        self._pause_evt.set()

    def is_paused(self) -> bool:
        return not self._pause_evt.is_set()

    def stop(self):
        self._stop_evt.set()
        self._pause_evt.set()


# ----------------------------------------------------------------------
# Recorded-motion txt parser
# ----------------------------------------------------------------------
def parse_motion_file(path, expected_len: int = 20):
    """Parse `[v0,...,vN][v0,...,vN]...` → list of length-`expected_len`
    int lists. Frames whose length differs are skipped (with a warning)."""
    text = Path(path).read_text()
    frames = []
    skipped = 0
    for m in re.finditer(r'\[([^\]]+)\]', text):
        body = m.group(1)
        try:
            values = [int(v.strip()) for v in body.split(',')]
        except ValueError:
            skipped += 1
            continue
        if len(values) != expected_len:
            skipped += 1
            continue
        frames.append(values)
    if skipped:
        print(f"[parser] {skipped} frame(s) skipped (wrong length or non-int).")
    return frames
