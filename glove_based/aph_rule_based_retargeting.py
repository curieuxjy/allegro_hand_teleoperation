#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
Manus Glove to APH Hand Rule-Based Retargeting — Simulator Verification

Parallel to v6_rule_based_retargeting.py but for the aph_right hand
(16 DOF, 4 fingers — Thumb, Index, Middle, Ring; no pinky).

Shared bits (Sapien helpers, ErgonomicsBarViz, TuningSliders, tuning IO)
live in retargeting_utils. This file keeps the aph-specific math and
config: joint limits, default scales/offsets, the ergonomics normalization
table, the `transform_glove_to_aph` mapping, and the slider/viz layout.

aph URDF (see glove_based/assets/aph_right/aph_right.urdf) summary:
  Thumb   j00 [-100°, 0°]  CMC abduction (Z-axis, entirely negative)
  Thumb   j01 [-20°, +67°] CMC flex      (X-axis, mostly positive)
  Thumb   j02 [0°, +67°]   MCP flex      (-Y axis, entirely positive)
  Thumb   j03 [0°, +90°]   IP  flex      (-Y axis, entirely positive)
  Non-thumb jN0 [±30°]     spread        (Y-axis, symmetric)
  Non-thumb jN1..jN3 [-67°, 0°]          flex encoded NEGATIVE

NOTE: this script expects a config file at
  glove_based/geort/config/aph_right.json
defining `joint_order` (URDF order: Thumb → Index → Middle → Ring) and
`fingertip_link`. Create it before running.

CLI (mirrors v6 script)
  --hand          : config name (default aph_right)
  --side          : glove side to subscribe to (default right)
  --no-kinematic  : opt out of kinematic snap mode (default ON)
  --no-tune       : opt out of the live tuning slider window (default ON)
  --no-ergo-viz   : opt out of the Open3D ergonomics window
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
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from geort.utils.config_utils import get_config
    from geort.env.hand_debug import HandKinematicModel, HandViewerEnv
    import retargeting_utils as ru


# Joint limits for aph_right (radians), URDF order:
#   Thumb (joint_00..03) → Index (joint_10..13) →
#   Middle (joint_20..23) → Ring (joint_30..33).
APH_UPPER_LIMITS = np.array([
     0.0,        1.1693706,  1.1693706,  1.5707963,  # thumb (00, 01, 02, 03)
     0.5235988,  0.0,        0.0,        0.0,         # index (10, 11, 12, 13)
     0.5235988,  0.0,        0.0,        0.0,         # middle
     0.5235988,  0.0,        0.0,        0.0,         # ring
])

APH_LOWER_LIMITS = np.array([
    -1.7453293, -0.3490659,  0.0,        0.0,         # thumb
    -0.5235988, -1.1693706, -1.1693706, -1.1693706,   # index
    -0.5235988, -1.1693706, -1.1693706, -1.1693706,   # middle
    -0.5235988, -1.1693706, -1.1693706, -1.1693706,   # ring
])


# Per-finger colors (no pinky entry — aph has 4 fingers).
APH_FINGER_RGBA = {
    'thumb':  [0.95, 0.30, 0.30, 1.0],   # red
    'index':  [0.30, 0.55, 0.95, 1.0],   # blue
    'middle': [0.30, 0.85, 0.40, 1.0],   # green
    'ring':   [0.95, 0.65, 0.20, 1.0],   # orange
}
APH_FINGER_RGB = {k: v[:3] for k, v in APH_FINGER_RGBA.items()}


def _aph_link_classifier(link_name: str):
    """Map an aph URDF link name to its finger color (or None).
    Link names follow `<finger>_<n>` (e.g. thumb_0, index_3, ring_tip)."""
    for finger, rgba in APH_FINGER_RGBA.items():
        if link_name.startswith(finger + '_'):
            return rgba
    return None


# ----------------------------------------------------------------------
# TuningSliders layout for aph (4 fingers × Step 2/Step 3)
# ----------------------------------------------------------------------
APH_SCALE_LAYOUT = [
    ('Thumb', [
        ('t00', '00 (abduct)'),
        ('t01', '01 (CMC flex)'),
        ('t02', '02 (MCP)'),
        ('t03', '03 (IP)'),
    ]),
    ('Index', [
        # i10 (IndexMCPSpread) is dropped by the Manus publisher;
        # j10 is synthesized at Step 2 from middle + ring spread.
        ('i_from_m', '10 ← middle'),
        ('i_from_r', '10 ← ring'),
        ('i11', '11 (MCP)'),
        ('i12', '12 (PIP)'),
        ('i13', '13 (DIP)'),
    ]),
    ('Middle', [
        ('m20', '20 (spread)'), ('m21', '21 (MCP)'),
        ('m22', '22 (PIP)'),    ('m23', '23 (DIP)'),
    ]),
    ('Ring', [
        ('r30', '30 (spread)'), ('r31', '31 (MCP)'),
        ('r32', '32 (PIP)'),    ('r33', '33 (DIP)'),
    ]),
]
APH_OFFSET_LAYOUT = [
    ('Thumb',  ['o_t00', 'o_t01', 'o_t02', 'o_t03']),
    ('Index',  ['o_i10', 'o_i11', 'o_i12', 'o_i13']),
    ('Middle', ['o_m20', 'o_m21', 'o_m22', 'o_m23']),
    ('Ring',   ['o_r30', 'o_r31', 'o_r32', 'o_r33']),
]
APH_REVERSED_SCALE_KEYS = set()   # no reversed sliders for aph by default


class ManusAphSimNode(Node):
    """ROS2 subscriber: turns Manus glove messages into aph joint angles.

    Stores the latest 16-D qpos (URDF order: thumb→index→middle→ring, rad)
    in `self.latest_qpos` for the main thread to consume. Tunable
    constants live in `self.scales` (Step 2 multipliers) and
    `self.offsets` (Step 3 biases in degrees); dict mutation is atomic in
    CPython so the slider thread can update them live without a lock.
    """

    # All scales stored as POSITIVE magnitudes. The sign of each
    # contribution is hardcoded in `transform_glove_to_aph` based on the
    # URDF joint range (e.g. j00 has entirely negative range so the
    # transform applies `-n('t00', ...)`).
    DEFAULT_SCALES = {
        # Thumb
        't00': 1.0, 't01': 1.0, 't02': 1.0, 't03': 1.0,
        # Index (i10 is omitted — IndexMCPSpread is dropped by the
        # Manus publisher and j10 is synthesized in Step 2 from middle
        # + ring spread terms; see i_from_m / i_from_r below.)
        'i11': 1.0, 'i12': 1.0, 'i13': 1.0,
        # Middle
        'm20': 1.0, 'm21': 1.0, 'm22': 1.0, 'm23': 1.0,
        # Ring
        'r30': 1.0, 'r31': 1.0, 'r32': 1.0, 'r33': 1.0,
        # IndexMCPSpread synthesis weights — j10 is mixed from middle
        # (m20 term) and ring (r30 term) in Step 2. Default 0.5/0.5.
        'i_from_m': 0.5, 'i_from_r': 0.5,
    }

    DEFAULT_OFFSETS = {
        'o_t00': 0.0, 'o_t01': 0.0, 'o_t02': 0.0, 'o_t03': 0.0,
        'o_i10': 0.0, 'o_i11': 0.0, 'o_i12': 0.0, 'o_i13': 0.0,
        'o_m20': 0.0, 'o_m21': 0.0, 'o_m22': 0.0, 'o_m23': 0.0,
        'o_r30': 0.0, 'o_r31': 0.0, 'o_r32': 0.0, 'o_r33': 0.0,
    }

    TUNING_FILE = Path(__file__).resolve().parent / 'aph_tuning.json'
    RECORDING_DIR = Path(__file__).resolve().parent / 'recordings'

    DEG_TO_COUNTS = 4096.0 / 360.0

    # Normalization table: per-term contribution =
    #   scale × (ergo_val / ERGO_MAX[type]) × joint_amp
    # so ergo=0 → 0 deg and ergo at typical max → scale × joint_amp deg.
    # joint_amp is the available motion amplitude on each aph joint
    # (read off APH_LOWER/UPPER_LIMITS).
    ERGO_MAX_DEG = {'spread': 30.0, 'stretch': 70.0}
    SCALE_NORM_INFO = {
        # Thumb — j00 100° one-sided, j01 ~67° (positive half),
        # j02 67°, j03 90°
        't00': (100.0, 'spread'),
        't01': (67.0,  'stretch'),
        't02': (67.0,  'stretch'),
        't03': (90.0,  'stretch'),
        # Index — j10 synthesized in Step 2 (no _norm entry needed);
        # j11..j13 67° magnitude
        'i11': (67.0,  'stretch'),
        'i12': (67.0,  'stretch'),
        'i13': (67.0,  'stretch'),
        # Middle
        'm20': (30.0,  'spread'),
        'm21': (67.0,  'stretch'),
        'm22': (67.0,  'stretch'),
        'm23': (67.0,  'stretch'),
        # Ring
        'r30': (30.0,  'spread'),
        'r31': (67.0,  'stretch'),
        'r32': (67.0,  'stretch'),
        'r33': (67.0,  'stretch'),
    }

    # Ergonomics parser layout — pinky entries from the 5-finger glove
    # are silently dropped (no Pinky key in _FINGER_OFFSET).
    _FINGER_OFFSET = {'Thumb': 0, 'Index': 4, 'Middle': 8, 'Ring': 12}
    _MOTION_IDX = {
        'MCPSpread':  0, 'Spread': 0,
        'MCPStretch': 1, 'PIPStretch': 2, 'DIPStretch': 3,
    }

    def __init__(self, side: str = 'right'):
        super().__init__('manus_aph_sim_node')
        self.side = side.lower()

        self.alpha = 0.2          # EMA smoothing coefficient
        self.prev_arr = None

        self.scales = dict(self.DEFAULT_SCALES)
        self.offsets = dict(self.DEFAULT_OFFSETS)
        ru.load_tuning_file(self.TUNING_FILE, self.scales, self.offsets,
                            logger=self.get_logger())

        self.latest_qpos = None       # 16-D aph joint angles (rad, URDF order)
        self.latest_glove16 = None    # 16-D raw glove ergonomics (deg)
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
        timestamped .txt. No sign-flips on save (unlike V6 — aph motor
        convention is not assumed; add flips here if/when needed for
        replay-to-hardware). Returns the saved Path or None."""
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
            hz_tag = f'{self._rec_target_hz:g}hz'
            path = self.RECORDING_DIR / f'aph_motion_{ts}_{hz_tag}.txt'
            with open(path, 'w') as f:
                for frame in frames:
                    counts = np.round(
                        np.rad2deg(frame) * self.DEG_TO_COUNTS
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
        ru.reload_tuning_in_place(
            self.TUNING_FILE,
            self.scales, self.offsets,
            self.DEFAULT_SCALES, self.DEFAULT_OFFSETS,
            logger=self.get_logger(),
        )

    # ----- Glove → aph mapping ----------------------------------------
    def _norm(self, key, ergo_val):
        amp, ergo_type = self.SCALE_NORM_INFO[key]
        return self.scales[key] * (ergo_val / self.ERGO_MAX_DEG[ergo_type]) * amp

    def transform_glove_to_aph(self, glove16):
        """
        16-D Manus glove ergonomics → 16-D aph joint angles
        (URDF order: Thumb → Index → Middle → Ring; no pinky).

        Glove channel layout per finger (per ERGONOMICS.md §2):
            val[0] = {Finger}Spread / ThumbMCPSpread     — abduction (deg)
            val[1] = {Finger}MCPStretch / ThumbMCPStretch — MCP / CMC flex
            val[2] = {Finger}PIPStretch / ThumbPIPStretch — PIP / MCP flex
            val[3] = {Finger}DIPStretch / ThumbDIPStretch — DIP / IP  flex
        Per ERGONOMICS.md §3, "*Stretch" = flexion (+ = curl into palm).

        RIGHT-HAND ONLY for spread signs. ERGONOMICS.md §3 documents that
        non-thumb Spread channels mirror between left and right; every
        spread mapping below must be re-checked for --side=left.

        Design differences vs V6 transform:
          * 4 fingers (no pinky) — 16 DOFs total.
          * Thumb joints SEPARATED: j00 = CMC abduction (val[0]) only,
            j01 = CMC flex (val[1]) only. V6 mixes both onto j00.
          * Non-thumb flex joints (jN1..jN3) encode flexion as NEGATIVE
            joint values (URDF range [-67°, 0°]) → sign-flip + flex.
          * No mirror baseline on j01 (mixed range, direct mapping).
          * No V6-style pinky↔ring anti-collision (no pinky).

        Pipeline:
          1. slice per-finger glove values
          2. normalized scale (j10 index spread synthesized inline here
             from middle + ring terms via `i_from_m` / `i_from_r`)
          3. additive per-joint offset (deg)
          4. deg → rad
          5. clip to APH_LOWER/UPPER_LIMITS (also discards hyperextension)
          6. EMA smoothing
        """
        # Step 1 — per-finger 4-vectors [Spread, MCP, PIP, DIP] in degrees.
        # index_vals[0] is always 0 because the Manus publisher drops
        # IndexMCPSpread (ErgonomicsDataTypeToSide bug). Step 7 fills it.
        thumb_vals  = np.array(glove16[0:4],   dtype=float)
        index_vals  = np.array(glove16[4:8],   dtype=float)
        middle_vals = np.array(glove16[8:12],  dtype=float)
        ring_vals   = np.array(glove16[12:16], dtype=float)

        # Step 2 — normalized scale terms. Per-line sign chosen from URDF
        # joint range; joint-range annotations below are in degrees.
        n = self._norm
        s = self.scales
        angle_deg = np.array([
            # ── Thumb (joint_00..03) ────────────────────────────────
            # j00 [-100°, 0°]: CMC abduction (Z-axis, entirely NEGATIVE
            # range). Manus + ThumbMCPSpread = palmar abduction → aph
            # joint must go negative → negate the term.
            -n('t00', thumb_vals[0]),
            # j01 [-20°, +67°]: CMC flex (X-axis, mixed range, mostly
            # positive). Manus + ThumbMCPStretch = CMC flex → + joint.
            # Direct mapping.
            n('t01', thumb_vals[1]),
            # j02 [0°, +67°]: MCP flex (-Y axis, entirely POSITIVE).
            # Direct + → +.
            n('t02', thumb_vals[2]),
            # j03 [0°, +90°]: IP flex (-Y axis, entirely POSITIVE).
            # Direct + → +.
            n('t03', thumb_vals[3]),
            # ── Index (joint_10..13) ────────────────────────────────
            # j10 [±30°]: spread (Y-axis, symmetric). IndexMCPSpread is
            # dropped by the Manus publisher (val[0] always 0), so j10
            # is synthesized here from the SAME middle (m20) and ring
            # (r30) normalized terms used below, mixed via tunable
            # weights `i_from_m` / `i_from_r` (default 0.5 / 0.5).
            (s['i_from_m'] * n('m20', middle_vals[0])
             + s['i_from_r'] * n('r30', ring_vals[0])),
            # j11..j13 [-67°, 0°]: flex encoded as NEGATIVE on aph
            # (entirely negative joint range, opposite sign of V6/glove).
            # + Manus flex → must produce − joint → negate.
            -n('i11', index_vals[1]),
            -n('i12', index_vals[2]),
            -n('i13', index_vals[3]),
            # ── Middle (joint_20..23) ───────────────────────────────
            # j20 [±30°]: spread (Y-axis, symmetric). Middle spread
            # polarity is ambiguous in ERGONOMICS.md; direct mapping
            # is a starting point — flip via slider if mirrored.
            n('m20', middle_vals[0]),
            -n('m21', middle_vals[1]),   # j21 [-67°, 0°]: flex negated
            -n('m22', middle_vals[2]),   # j22 [-67°, 0°]: flex negated
            -n('m23', middle_vals[3]),   # j23 [-67°, 0°]: flex negated
            # ── Ring (joint_30..33) ─────────────────────────────────
            # j30 [±30°]: spread (Y-axis). aph ring base is at -X; +Y
            # joint rotation moves the tip toward +X (toward middle =
            # radial). Anatomical splay outward is ulnar (away from
            # middle, toward where pinky would be) = -X = − joint.
            # Manus right-hand ring is − when splayed ulnar →
            # direct mapping − → − is consistent.
            n('r30', ring_vals[0]),
            -n('r31', ring_vals[1]),     # j31..j33: flex negated
            -n('r32', ring_vals[2]),
            -n('r33', ring_vals[3]),
        ], dtype=float)

        # Step 3 — additive per-joint offsets (deg). Sliders mutate
        # self.offsets live. Defaults are 0 — most aph joints have
        # ranges that include 0°, so no large offsets are required up
        # front (unlike Allegro's thumb).
        o = self.offsets
        offset_deg = np.array([
            o['o_t00'], o['o_t01'], o['o_t02'], o['o_t03'],
            o['o_i10'], o['o_i11'], o['o_i12'], o['o_i13'],
            o['o_m20'], o['o_m21'], o['o_m22'], o['o_m23'],
            o['o_r30'], o['o_r31'], o['o_r32'], o['o_r33'],
        ], dtype=float)
        angle_deg = angle_deg + offset_deg

        # Steps 4-5 — deg → rad, clip to aph limits. The clip also
        # discards glove-side hyperextension per ERGONOMICS.md §4 item 3.
        arr = np.deg2rad(angle_deg)
        arr = np.clip(arr, APH_LOWER_LIMITS, APH_UPPER_LIMITS)

        # Step 6 — EMA smoothing.
        if self.prev_arr is None:
            smoothed = arr.copy()
        else:
            smoothed = self.alpha * arr + (1.0 - self.alpha) * self.prev_arr
        self.prev_arr = smoothed
        return smoothed

    def _ergonomics_to_array(self, ergonomics_list):
        """Parse Manus ergonomics list into a 16-D array in URDF order
        (thumb → index → middle → ring). Pinky entries are ignored.
        IndexMCPSpread is dropped by the publisher and stays 0."""
        glove16 = np.zeros(16, dtype=float)
        for ergo in ergonomics_list:
            t = ergo.type
            for finger, offset in self._FINGER_OFFSET.items():
                if t.startswith(finger):
                    motion = t[len(finger):]
                    if motion in self._MOTION_IDX:
                        glove16[offset + self._MOTION_IDX[motion]] = ergo.value
                    break
        return glove16

    def glove_callback(self, msg: ManusGlove):
        if len(msg.ergonomics) == 0:
            return
        glove16 = self._ergonomics_to_array(msg.ergonomics)
        self.latest_glove16 = glove16
        self.latest_qpos = self.transform_glove_to_aph(glove16)

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
        description='aph hand rule-based retargeting (simulator verification).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: kinematic snap + live tuning sliders, /manus_glove_right
  python aph_rule_based_retargeting.py

  # Physics-mode PD drive instead of kinematic snap
  python aph_rule_based_retargeting.py --no-kinematic

  # Skip the slider window
  python aph_rule_based_retargeting.py --no-tune

  # Left glove (requires aph_left config to exist)
  python aph_rule_based_retargeting.py --hand aph_left --side left
        """
    )
    parser.add_argument('--hand', type=str, default='aph_right',
                        help='Config name to load (default: aph_right)')
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
    print(f"[aph_rule_based_retargeting] loading config: {config['name']}")

    model = HandKinematicModel.build_from_config(config, scene=scene, render=False)
    model.hand.set_root_pose(sapien.Pose([0.0, 0.0, 0.35], [0.695, 0.0, -0.718, 0.0]))
    ru.colorize_articulation(model.hand, _aph_link_classifier)

    if not args.kinematic:
        ru.tune_pd_drives(model)

    links = [info['link'] for info in config['fingertip_link']]
    offsets = [info['center_offset'] for info in config['fingertip_link']]
    model.initialize_keypoint(links, offsets)
    viewer_env = HandViewerEnv([model], scene=scene, renderer=renderer)

    # Optional ergonomics debug window (16 channels for 4 fingers)
    ergo_viz = None
    if not args.no_ergo_viz:
        ergo_viz = ru.ErgonomicsBarViz(
            APH_LOWER_LIMITS, APH_UPPER_LIMITS,
            fingers=['thumb', 'index', 'middle', 'ring'],
            finger_rgb=APH_FINGER_RGB,
            window_title='Manus ergonomics (front) / aph joint fraction (back)',
            window_size=(1000, 620),
        )

    # ROS2 init + background spin
    rclpy.init()
    node = ManusAphSimNode(side=args.side)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Optional tuner
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
                with_load=True, with_record=True, default_record_hz=120,
                with_broadcast=False,
                title='aph Retargeting Tuning',
            )
            print("[aph_rule_based_retargeting] tuning slider window open.")
        except RuntimeError as e:
            print(f"[aph_rule_based_retargeting] failed to open tuner: {e}")

    mode_str = 'kinematic snap' if args.kinematic else 'physics / PD'
    print(f"[aph_rule_based_retargeting] running in {mode_str} mode. "
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
                ergo_viz.update(node.latest_glove16, node.latest_qpos)
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
