#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
Manus Glove to V6 Hand Rule-Based Retargeting — Simulator Verification

ROS2 subscriber → glove→qpos pipeline → Sapien viewer + Open3D ergo bars
+ tkinter tuning sliders + motion-recording-to-txt. Shared bits (Sapien
helpers, ErgonomicsBarViz, TuningSliders, tuning JSON IO) live in
retargeting_utils. This file keeps only the V6-specific math and config:
joint limits, default scales/offsets, the ergonomics normalization table,
the `transform_glove_to_v6` mapping, and the slider/viz layout.

CLI
  --hand          : config name (default v6_right)
  --side          : glove side to subscribe to (default right)
  --no-kinematic  : opt out of kinematic snap mode (default ON)
  --no-tune       : opt out of the live tuning slider window (default ON)
  --no-ergo-viz   : opt out of the Open3D ergonomics / joint-fraction window
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

# Path setup for direct script execution
try:
    from .geort.utils.config_utils import get_config
    from .geort.env.hand_debug import HandKinematicModel, HandViewerEnv
    from . import retargeting_utils as ru
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from geort.utils.config_utils import get_config
    from geort.env.hand_debug import HandKinematicModel, HandViewerEnv
    import retargeting_utils as ru


# Joint limits for V6 right hand (radians), in URDF/config user order:
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


# Per-finger colors used for both Sapien link tint and ergo bar tint.
V6_FINGER_RGBA = {
    'thumb':  [0.95, 0.30, 0.30, 1.0],   # red
    'index':  [0.30, 0.55, 0.95, 1.0],   # blue
    'middle': [0.30, 0.85, 0.40, 1.0],   # green
    'ring':   [0.95, 0.65, 0.20, 1.0],   # orange
    'pinky':  [0.75, 0.40, 0.85, 1.0],   # purple
}
V6_FINGER_RGB = {k: v[:3] for k, v in V6_FINGER_RGBA.items()}


def _v6_link_classifier(link_name: str):
    """Map a V6 URDF link name to its finger color (or None)."""
    for finger, rgba in V6_FINGER_RGBA.items():
        if link_name.startswith(finger + '_'):
            return rgba
    return None


# ----------------------------------------------------------------------
# TuningSliders layout for V6 (5 fingers × Step 2/Step 3)
# ----------------------------------------------------------------------
V6_SCALE_LAYOUT = [
    ('Thumb', [
        ('t00_curl',   '00 ← curl'),
        ('t00_spread', '00 ← spread'),
        ('t01',        '01 (curl)'),
        ('t02',        '02 (MCP)'),
        ('t03',        '03 (IP)'),
    ]),
    ('Index', [
        ('i10', '10 (spread)'), ('i11', '11 (MCP)'),
        ('i12', '12 (PIP)'),    ('i13', '13 (DIP)'),
    ]),
    ('Middle', [
        ('m20', '20 (spread)'), ('m21', '21 (MCP)'),
        ('m22', '22 (PIP)'),    ('m23', '23 (DIP)'),
    ]),
    ('Ring', [
        ('r30', '30 (spread)'), ('r31', '31 (MCP)'),
        ('r32', '32 (PIP)'),    ('r33', '33 (DIP)'),
    ]),
    ('Pinky', [
        ('p40_curl',   '40 ← curl'),
        ('p40_spread', '40 ← spread'),
        ('p41',        '41 (MCP)'),
        ('p42',        '42 (PIP)'),
        ('p43',        '43 (DIP)'),
    ]),
]
V6_OFFSET_LAYOUT = [
    ('Thumb',  ['o_t00', 'o_t01', 'o_t02', 'o_t03']),
    ('Index',  ['o_i10', 'o_i11', 'o_i12', 'o_i13']),
    ('Middle', ['o_m20', 'o_m21', 'o_m22', 'o_m23']),
    ('Ring',   ['o_r30', 'o_r31', 'o_r32', 'o_r33']),
    ('Pinky',  ['o_p40', 'o_p41', 'o_p42', 'o_p43']),
]
# Sliders whose physical direction is reversed (left = high, right = low).
V6_REVERSED_SCALE_KEYS = {'p40_curl'}


class ManusV6SimNode(Node):
    """ROS2 subscriber: turns Manus glove messages into V6 joint angles.

    Stores the latest 20-D qpos (user order, radians) in `self.latest_qpos`
    for the main thread to consume. Tunable constants live in `self.scales`
    (Step 2 multipliers) and `self.offsets` (Step 3 biases in degrees) —
    dict mutation is atomic in CPython so the slider thread can update them
    live without a lock.
    """

    # Step 2 default scales — all stored as POSITIVE magnitudes. The sign
    # of each contribution is hardcoded in `transform_glove_to_v6` (e.g.
    # the pinky term `-(... - ...)`). joint_00 and joint_40 use TWO entries
    # each because curl and spread are mixed.
    DEFAULT_SCALES = {
        # Thumb
        't00_curl': 1.0, 't00_spread': 1.0,
        't01': 1.0, 't02': 1.0, 't03': 1.0,
        # Index
        'i10': 1.0, 'i11': 1.0, 'i12': 1.0, 'i13': 1.0,
        # Middle
        'm20': 1.0, 'm21': 1.0, 'm22': 1.0, 'm23': 1.0,
        # Ring
        'r30': 1.0, 'r31': 1.0, 'r32': 1.0, 'r33': 1.0,
        # Pinky
        'p40_curl': 1.0, 'p40_spread': 1.0,
        'p41': 1.0, 'p42': 1.0, 'p43': 1.0,
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

    DEG_TO_COUNTS = 4096.0 / 360.0

    # Thumb joint_01 mirror baseline (range is one-sided [0, 160] deg, so
    # we anchor at the top and subtract the scaled term).
    T01_MIRROR_BASELINE_DEG = 160.0

    # Normalization table: per-term contribution =
    #   scale × (ergo_val / ERGO_MAX[type]) × joint_amp
    # so ergo=0 → 0 deg and ergo at typical max → scale × joint_amp deg.
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

    # Ergonomics parser layout. Manus drops IndexMCPSpread (publisher bug),
    # and labels the non-thumb spread channels as "MiddleSpread" rather
    # than "MiddleMCPSpread" — we accept both.
    _FINGER_OFFSET = {'Thumb': 0, 'Index': 4, 'Middle': 8, 'Ring': 12, 'Pinky': 16}
    _MOTION_IDX = {
        'MCPSpread':  0, 'Spread': 0,
        'MCPStretch': 1, 'PIPStretch': 2, 'DIPStretch': 3,
    }

    def __init__(self, side: str = 'right'):
        super().__init__('manus_v6_sim_node')
        self.side = side.lower()

        self.alpha = 0.2          # EMA smoothing coefficient
        self.prev_arr = None

        self.scales = dict(self.DEFAULT_SCALES)
        self.offsets = dict(self.DEFAULT_OFFSETS)
        ru.load_tuning_file(self.TUNING_FILE, self.scales, self.offsets,
                            logger=self.get_logger())

        self.latest_qpos = None       # 20-D V6 joint angles (rad, user order)
        self.latest_glove20 = None    # 20-D raw glove ergonomics (deg)
        self._got_first = False

        # Recording state (toggled from the tuner window)
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

    # ----- Recording ---------------------------------------------------
    def start_recording(self, target_hz: float = 120.0):
        target_hz = float(max(0.1, min(target_hz, 120.0)))
        self.recorded_frames = []
        self._rec_t0 = time.monotonic()
        self._rec_target_hz = target_hz
        self._rec_period = 1.0 / target_hz
        self._rec_next_t = None
        self.recording = True
        self.get_logger().info(
            f'Motion recording started @ target {target_hz:.1f} Hz.'
        )

    def stop_recording(self):
        """Stop recording and write frames as encoder counts to a
        timestamped .txt. Sign-flips thumb_00 and pinky_40 on disk so the
        file can be replayed directly to the motors. Returns the saved
        Path on success, or None if nothing was recorded."""
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
                    save_frame[0]  *= -1   # thumb joint_00 sign-flip
                    save_frame[16] *= -1   # pinky joint_40 sign-flip
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
    def save_tuning(self):
        return ru.save_tuning_file(self.TUNING_FILE, self.scales,
                                   self.offsets, logger=self.get_logger())

    def reload_tuning(self):
        """Reset to defaults then overlay disk values, in place — the
        slider window holds references to self.scales / self.offsets."""
        ru.reload_tuning_in_place(
            self.TUNING_FILE,
            self.scales, self.offsets,
            self.DEFAULT_SCALES, self.DEFAULT_OFFSETS,
            logger=self.get_logger(),
        )

    # ----- Glove → V6 mapping -----------------------------------------
    def _norm(self, key, ergo_val):
        amp, ergo_type = self.SCALE_NORM_INFO[key]
        return self.scales[key] * (ergo_val / self.ERGO_MAX_DEG[ergo_type]) * amp

    def transform_glove_to_v6(self, glove20):
        """
        20-D Manus glove ergonomics → 20-D V6 joint angles (URDF order).

        Glove channel layout per finger (`val[k]` below), per ERGONOMICS.md §2
        and `_ergonomics_to_array`:
            val[0] = {Finger}Spread / ThumbMCPSpread     — abduction (deg)
            val[1] = {Finger}MCPStretch / ThumbMCPStretch — MCP / CMC flex
            val[2] = {Finger}PIPStretch / ThumbPIPStretch — PIP / MCP flex
            val[3] = {Finger}DIPStretch / ThumbDIPStretch — DIP / IP  flex
        Per ERGONOMICS.md §3, "*Stretch" is actually flexion and + = curl into
        the palm; ThumbMCPSpread + = palmar abduction.

        RIGHT-HAND ONLY for spread signs. Per ERGONOMICS.md §3, the non-thumb
        Spread channels follow "+ = radial, − = ulnar" on the right glove
        (Index radial-positive, Pinky ulnar-negative on splay). The left
        glove mirrors this — every spread-channel mapping below must be
        re-checked / sign-flipped before using this transform with --side=left.

        Pipeline:
          1. slice per-finger glove values
          2. normalized scale: term_deg = scale × (ergo/ERGO_MAX) × amp
          3. additive per-joint offset (deg)
          4. deg → rad
          5. clip to V6_LOWER/UPPER_LIMITS (also discards hyperextension)
          6. EMA smoothing (prev_arr is stored constraint-compliant)
          7. anti-collision clamp + IndexMCPSpread synthesis
        """
        # Step 1 — per-finger 4-vectors in [spread, MCP, PIP, DIP] order.
        # index_vals[0] is always 0 because the Manus publisher drops
        # IndexMCPSpread (ErgonomicsDataTypeToSide bug). Step 7b fills it.
        thumb_vals  = np.array(glove20[0:4],   dtype=float)
        index_vals  = np.array(glove20[4:8],   dtype=float)
        middle_vals = np.array(glove20[8:12],  dtype=float)
        ring_vals   = np.array(glove20[12:16], dtype=float)
        pinky_vals  = np.array(glove20[16:20], dtype=float)

        # Step 2 — normalized scale terms.
        # Joint range annotations below are deg from V6_LOWER/UPPER_LIMITS
        # and explain WHY each per-joint sign was chosen.
        n = self._norm
        angle_deg = np.array([
            # ── Thumb ────────────────────────────────────────────────
            # j00 [±80°]: V6 puts CMC flexion AND CMC abduction on the
            # same base-swing joint, so both terms add (both glove inputs
            # pull the thumb base in the same physical direction).
            n('t00_curl', thumb_vals[1]) + n('t00_spread', thumb_vals[0]),
            # j01 [0..160°]: thumb opposition. One-sided positive range
            # anchored at BASELINE = rest pose; larger CMC flex (val[1])
            # mirrors the joint downward (BASELINE − term).
            self.T01_MIRROR_BASELINE_DEG - n('t01', thumb_vals[1]),
            n('t02', thumb_vals[2]),  # j02 [-5..90°]: MCP flex (ThumbPIPStretch)
            n('t03', thumb_vals[3]),  # j03 [-5..90°]: IP  flex (ThumbDIPStretch)
            # ── Index ────────────────────────────────────────────────
            # j10 [-10..90°]: MCP spread. IndexMCPSpread is dropped by
            # the Manus publisher so val[0] is always 0 → this line
            # evaluates to 0 and is overwritten by Step 7b below.
            n('i10', index_vals[0]),
            n('i11', index_vals[1]),  # j11 [0..100°]: MCP flex
            n('i12', index_vals[2]),  # j12 [-5..90°]: PIP flex
            n('i13', index_vals[3]),  # j13 [-5..90°]: DIP flex
            # ── Middle ───────────────────────────────────────────────
            n('m20', middle_vals[0]),  # j20 [±80°]: spread (symmetric)
            n('m21', middle_vals[1]),  # j21 [0..100°]: MCP flex
            n('m22', middle_vals[2]),  # j22 [-5..90°]: PIP flex
            n('m23', middle_vals[3]),  # j23 [-5..90°]: DIP flex
            # ── Ring ─────────────────────────────────────────────────
            # j30 [-80..+10°]: spread biased negative because ring's
            # anatomical splay is ulnar (toward pinky). On the right
            # hand that's a negative Manus Spread, so direct mapping
            # − → − is consistent. (Left hand: invert sign.)
            n('r30', ring_vals[0]),
            n('r31', ring_vals[1]),  # j31 [0..100°]: MCP flex
            n('r32', ring_vals[2]),  # j32 [-5..90°]: PIP flex
            n('r33', ring_vals[3]),  # j33 [-5..90°]: DIP flex
            # ── Pinky ────────────────────────────────────────────────
            # j40 [-100..0°]: pinky base swings only ulnar on V6 (range
            # entirely negative). Both inputs must push joint negative:
            #   • curl_term   from val[1] (+ = flex)  →  -curl_term  is −
            #   • spread_term from val[0] (− when splayed ulnar, right
            #     hand)                                →  +spread_term is −
            # Combined as -(curl − spread) = -curl + spread.
            -(n('p40_curl', pinky_vals[1]) - n('p40_spread', pinky_vals[0])),
            # j41 [-90..+10°]: MCP-like, but its flexion is encoded
            # NEGATIVE on V6 (look at the range). Sign-flip the +flex
            # glove value so + curl glove → − joint.
            -n('p41', pinky_vals[1]),
            n('p42', pinky_vals[2]),  # j42 [-5..90°]: PIP flex
            n('p43', pinky_vals[3]),  # j43 [-5..90°]: DIP flex
        ], dtype=float)

        # Step 3 — additive offsets (deg). Sliders mutate self.offsets live.
        o = self.offsets
        offset_deg = np.array([
            o['o_t00'], o['o_t01'], o['o_t02'], o['o_t03'],
            o['o_i10'], o['o_i11'], o['o_i12'], o['o_i13'],
            o['o_m20'], o['o_m21'], o['o_m22'], o['o_m23'],
            o['o_r30'], o['o_r31'], o['o_r32'], o['o_r33'],
            o['o_p40'], o['o_p41'], o['o_p42'], o['o_p43'],
        ], dtype=float)
        angle_deg = angle_deg + offset_deg

        # Steps 4-5 — deg → rad, clip to V6 limits. The clip also discards
        # glove-side hyperextension (most V6 joints stop at ~−5°),
        # matching ERGONOMICS.md §4 item 3.
        arr = np.deg2rad(angle_deg)
        arr = np.clip(arr, V6_LOWER_LIMITS, V6_UPPER_LIMITS)

        # Step 6 — EMA smoothing. We store the post-Step-7 result so the
        # next frame's blend starts from a constraint-compliant state.
        if self.prev_arr is None:
            smoothed = arr.copy()
        else:
            smoothed = self.alpha * arr + (1.0 - self.alpha) * self.prev_arr

        # Step 7a — V6 geometry anti-collision: pinky j41 (idx 17,
        # MCP-like flex) must stay ≤ ring j30 (idx 12, spread). The two
        # are different axis types; this is an empirical clamp on V6
        # link geometry, not a physiological constraint. When ring
        # abducts toward pinky (j30 negative), pinky flex must be at
        # least as negative or the links overlap.
        if smoothed[17] > smoothed[12]:
            smoothed[17] = smoothed[12]

        # Step 7b — synthesize IndexMCPSpread. The Manus publisher drops
        # the index spread channel, so smoothed[4] is 0 from Step 2. We
        # approximate index spread as 1.2 × middle spread (smoothed[8])
        # on the assumption that during a hand-splay gesture, index
        # tracks middle in the same direction with slightly larger
        # amplitude. Empirical multiplier — re-tune if needed.
        smoothed[4] = smoothed[8] * 1.2

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

        # Drift-corrected recording scheduler.
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
                f'First glove command received ({len(msg.ergonomics)} entries); '
                f'driving sim.'
            )


def main():
    parser = argparse.ArgumentParser(
        description='V6 hand rule-based retargeting (simulator verification).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: kinematic snap + live tuning sliders, /manus_glove_right
  python v6_rule_based_retargeting.py

  # Physics-mode PD drive instead of kinematic snap
  python v6_rule_based_retargeting.py --no-kinematic

  # Skip the slider window
  python v6_rule_based_retargeting.py --no-tune

  # Left glove
  python v6_rule_based_retargeting.py --hand v6_left --side left
        """
    )
    parser.add_argument('--hand', type=str, default='v6_right',
                        help='Config name to load (default: v6_right)')
    parser.add_argument('--side', type=str, choices=['left', 'right'], default='right',
                        help='Glove side to subscribe to (default: right)')
    parser.add_argument('--no-kinematic', action='store_false', dest='kinematic',
                        help='Physics-mode PD drive (default is kinematic snap).')
    parser.add_argument('--no-ergo-viz', action='store_true',
                        help='Disable the Open3D ergonomics window.')
    parser.add_argument('--no-tune', action='store_false', dest='tune',
                        help='Disable the tkinter live tuning slider window.')
    args = parser.parse_args()

    # Sapien scene + hand model + viewer
    engine, renderer, scene = ru.setup_sapien_scene()
    config = get_config(args.hand)
    print(f"[v6_rule_based_retargeting] loading config: {config['name']}")

    model = HandKinematicModel.build_from_config(config, scene=scene, render=False)
    model.hand.set_root_pose(sapien.Pose([0.0, 0.0, 0.35], [0.695, 0.0, -0.718, 0.0]))
    ru.colorize_articulation(model.hand, _v6_link_classifier)

    if not args.kinematic:
        ru.tune_pd_drives(model)

    links = [info['link'] for info in config['fingertip_link']]
    offsets = [info['center_offset'] for info in config['fingertip_link']]
    model.initialize_keypoint(links, offsets)
    viewer_env = HandViewerEnv([model], scene=scene, renderer=renderer)

    # Optional ergonomics debug window
    ergo_viz = None
    if not args.no_ergo_viz:
        ergo_viz = ru.ErgonomicsBarViz(
            V6_LOWER_LIMITS, V6_UPPER_LIMITS,
            fingers=['thumb', 'index', 'middle', 'ring', 'pinky'],
            finger_rgb=V6_FINGER_RGB,
            window_title='Manus ergonomics (front) / V6 joint fraction (back)',
        )

    # ROS2 init + background spin
    rclpy.init()
    node = ManusV6SimNode(side=args.side)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Optional tuner
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
                with_load=True, with_record=True, default_record_hz=50,
                with_broadcast=False,
                title='V6 Retargeting Tuning',
            )
            print("[v6_rule_based_retargeting] tuning slider window open.")
        except RuntimeError as e:
            print(f"[v6_rule_based_retargeting] failed to open tuner: {e}")

    mode_str = 'kinematic snap' if args.kinematic else 'physics / PD'
    print(f"[v6_rule_based_retargeting] running in {mode_str} mode. "
          f"Waiting for /manus_glove_{args.side} ...")

    try:
        while True:
            qpos = node.latest_qpos
            if qpos is not None:
                ru.apply_qpos(model, qpos, args.kinematic)

            if args.kinematic:
                viewer_env._update_tip_positions()
                viewer_env._update_axis_markers()
                viewer_env.scene.update_render()
                viewer_env.viewer.render()
            else:
                viewer_env.update()

            if ergo_viz is not None:
                ergo_viz.update(node.latest_glove20, node.latest_qpos)
                ergo_viz.render()

            if tuner is not None:
                tuner.poll()
    except KeyboardInterrupt:
        pass
    finally:
        if tuner is not None:
            tuner.close()
        if ergo_viz is not None:
            ergo_viz.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
