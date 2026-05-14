#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
Manus Glove to Allegro Hand Rule-Based Retargeting — Simulator Verification

Like v6_rule_based_retargeting.py but for the Allegro 16-DOF hand
(4 fingers × 4 joints, no pinky). Shared bits (Sapien helpers,
ErgonomicsBarViz, TuningSliders, tuning JSON IO) live in retargeting_utils.
This file keeps only the Allegro-specific math and config: joint limits,
default scales/offsets, the `transform_glove_to_allegro` mapping (with the
val[0]↔val[1] thumb cross-channel), and the slider/viz layout.

CLI
  --hand          : config name (default allegro_right)
  --side          : glove side to subscribe to (default right)
  --no-kinematic  : opt out of kinematic snap (default ON)
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


# Joint limits for Allegro right hand (radians), in sim joint order:
# allegro_right.json's `joint_order` is [joint_0..3, joint_4..7,
# joint_8..11, joint_12..15] = Index → Middle → Ring → Thumb.
ALLEGRO_UPPER_LIMITS = np.array([
     0.4700,  1.6100,  1.7090,  1.6180,   # index  (joint_0..3)
     0.4700,  1.6100,  1.7090,  1.6180,   # middle (joint_4..7)
     0.4700,  1.6100,  1.7090,  1.6180,   # ring   (joint_8..11)
     1.3960,  1.1630,  1.6440,  1.7190,   # thumb  (joint_12..15)
])
ALLEGRO_LOWER_LIMITS = np.array([
    -0.4700, -0.1960, -0.1740, -0.2270,
    -0.4700, -0.1960, -0.1740, -0.2270,
    -0.4700, -0.1960, -0.1740, -0.2270,
     0.7000,  0.3000, -0.1890, -0.1620,
])


# Per-finger RGBA used for both link tint and ergo bar tint.
ALLEGRO_FINGER_RGBA = {
    'thumb':  [0.95, 0.30, 0.30, 1.0],   # red    (link_12..15)
    'index':  [0.30, 0.55, 0.95, 1.0],   # blue   (link_0..3)
    'middle': [0.30, 0.85, 0.40, 1.0],   # green  (link_4..7)
    'ring':   [0.95, 0.65, 0.20, 1.0],   # orange (link_8..11)
}
ALLEGRO_FINGER_RGB = {k: v[:3] for k, v in ALLEGRO_FINGER_RGBA.items()}

# Allegro URDF uses link_N.0 / link_N.0_tip naming; map the "link_N" prefix
# (after stripping the trailing `.0` etc.) to its finger color.
_ALLEGRO_LINK_TO_FINGER = {
    **{f'link_{n}':  'index'  for n in range(0, 4)},
    **{f'link_{n}':  'middle' for n in range(4, 8)},
    **{f'link_{n}':  'ring'   for n in range(8, 12)},
    **{f'link_{n}':  'thumb'  for n in range(12, 16)},
}


def _allegro_link_classifier(link_name: str):
    base = link_name.split('.')[0] if link_name.startswith('link_') else link_name
    finger = _ALLEGRO_LINK_TO_FINGER.get(base)
    return ALLEGRO_FINGER_RGBA.get(finger) if finger else None


# ----------------------------------------------------------------------
# TuningSliders layout (sim joint order: Index → Middle → Ring → Thumb)
# ----------------------------------------------------------------------
ALLEGRO_SCALE_LAYOUT = [
    ('Index', [
        # i10 (IndexMCPSpread) is dropped by the Manus publisher;
        # joint_0 is synthesized at Step 2 from middle + ring spread.
        ('i_from_m', '0 ← middle'),
        ('i_from_r', '0 ← ring'),
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
    ('Thumb', [
        ('t00', '00 CMC ← v[1]'),
        ('t01', '01 base ← v[0]'),
        ('t02', '02 mid ← v[2]'),
        ('t03', '03 tip ← v[3]'),
    ]),
]
ALLEGRO_OFFSET_LAYOUT = [
    ('Index',  ['o_i10', 'o_i11', 'o_i12', 'o_i13']),
    ('Middle', ['o_m20', 'o_m21', 'o_m22', 'o_m23']),
    ('Ring',   ['o_r30', 'o_r31', 'o_r32', 'o_r33']),
    ('Thumb',  ['o_t00', 'o_t01', 'o_t02', 'o_t03']),
]


class ManusAllegroSimNode(Node):
    """ROS2 subscriber: turns Manus glove messages into Allegro joint angles.

    Allegro thumb convention (from legacy rule_based_retargeting.py):
      - joint_12 (CMC)        driven by glove val[1]  (NOT val[0])
      - joint_13 (base swing) driven by glove val[0]  (NOT val[1])
      - joint_14 (MCP-like)   driven by val[2]
      - joint_15 (IP-like)    driven by val[3]
    The val[0]↔val[1] swap is intentional — Manus thumb spread/CMC sensors
    map to Allegro thumb's CMC/base axes in this inverted way.
    """

    # Step 2 default scales. Thumb coefficients fold the legacy post-rad
    # multipliers (`*2.5`, `*2`) into a single pre-deg2rad scale.
    DEFAULT_SCALES = {
        # Thumb
        't00': -4.375,   # arr[12] (CMC, joint_12)  ← (-1.75 * v[1]) * 2.5
        't01':  6.0,     # arr[13] (base, joint_13) ← (3.0 * v[0]) * 2
        't02':  3.0,     # arr[14] (mid,  joint_14) ← 3.0 * v[2]
        't03':  2.0,     # arr[15] (tip,  joint_15) ← v[3] * 2
        # Index (i10 is omitted — IndexMCPSpread is dropped by the
        # Manus publisher and joint_0 is synthesized in Step 2 from
        # middle + ring spread terms; see i_from_m / i_from_r below.)
        'i11':  1.5, 'i12': 1.0, 'i13': 2.0,
        # Middle
        'm20': -0.2,
        'm21':  1.5, 'm22': 1.0, 'm23': 2.0,
        # Ring
        'r30':  0.1,
        'r31':  1.5, 'r32': 1.0, 'r33': 2.0,
        # IndexMCPSpread synthesis weights — joint_0 is mixed from
        # middle (m20 term) and ring (r30 term) in Step 2. Default
        # 0.5/0.5 → index sits halfway between middle and ring on splay.
        'i_from_m': 0.5, 'i_from_r': 0.5,
    }

    DEFAULT_OFFSETS = {
        'o_t00': 225.0, 'o_t01':   0.0, 'o_t02': -30.0, 'o_t03':  0.0,
        'o_i10':   0.0, 'o_i11':   0.0, 'o_i12':   0.0, 'o_i13':  0.0,
        'o_m20':  -4.0, 'o_m21':   0.0, 'o_m22':   0.0, 'o_m23':  0.0,
        'o_r30':   0.0, 'o_r31':   0.0, 'o_r32':   0.0, 'o_r33': 10.0,
    }

    TUNING_FILE = Path(__file__).resolve().parent / 'allegro_tuning.json'
    RECORDING_DIR = Path(__file__).resolve().parent / 'recordings'

    DEG_TO_COUNTS = 4096.0 / 360.0

    _FINGER_OFFSET = {'Index': 0, 'Middle': 4, 'Ring': 8, 'Thumb': 12}
    _MOTION_IDX = {
        'MCPSpread':  0, 'Spread': 0,
        'MCPStretch': 1, 'PIPStretch': 2, 'DIPStretch': 3,
    }

    def __init__(self, side: str = 'right'):
        super().__init__('manus_allegro_sim_node')
        self.side = side.lower()

        self.alpha = 0.2
        self.prev_arr = None

        self.scales = dict(self.DEFAULT_SCALES)
        self.offsets = dict(self.DEFAULT_OFFSETS)
        ru.load_tuning_file(self.TUNING_FILE, self.scales, self.offsets,
                            logger=self.get_logger())

        self.latest_qpos = None       # 16-D (rad, sim joint order)
        self.latest_glove16 = None    # 16-D raw glove ergonomics (deg)
        self._got_first = False

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
        """Stop and write frames to allegro_motion_<timestamp>.txt as
        bracketed encoder counts (16 ints per frame, sim joint order)."""
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
            path = self.RECORDING_DIR / f'allegro_motion_{ts}.txt'
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

    # ----- Glove → Allegro mapping ------------------------------------
    def transform_glove_to_allegro(self, glove16):
        """
        16-D Manus glove ergonomics → 16-D Allegro joint angles
        (sim joint order: Index → Middle → Ring → Thumb).

        Glove channel layout per finger (`val[k]` below), per ERGONOMICS.md §2
        and `_ergonomics_to_array`:
            val[0] = {Finger}Spread / ThumbMCPSpread     — abduction (deg)
            val[1] = {Finger}MCPStretch / ThumbMCPStretch — MCP / CMC flex
            val[2] = {Finger}PIPStretch / ThumbPIPStretch — PIP / MCP flex
            val[3] = {Finger}DIPStretch / ThumbDIPStretch — DIP / IP  flex
        Per ERGONOMICS.md §3, "*Stretch" is flexion and + = curl into palm.

        RIGHT-HAND ONLY for spread signs (per ERGONOMICS.md §3 the non-thumb
        Spread channels mirror between left and right; `--side=left` would
        require re-tuning every spread scale below).

        THUMB CROSS-CHANNEL SWAP (intentional, inherited from the legacy
        hardware-publishing rule_based_retargeting.py):
          joint_12 (CMC-like)  ← val[1]   (NOT val[0])
          joint_13 (base swing) ← val[0]  (NOT val[1])
        This deviates from the natural Spread→abduction / Stretch→flex
        mapping because Allegro's thumb joint axis convention is rotated
        relative to the Manus sensor frame on this finger.

        Other notes vs V6:
          * Pinky glove data is silently ignored (Allegro has no pinky).
          * Index spread is dropped by the Manus publisher → val[0] always 0;
            synthesized inline in Step 2 from middle (m20) + ring (r30)
            spread terms via tunable weights `i_from_m` / `i_from_r`
            (default 0.5 / 0.5).
          * No `_norm`/SCALE_NORM_INFO normalization — raw `scale × ergo_deg`.

        Pipeline: extract → scale (deg, with index j0 synthesized inline) →
                  offset (deg) → deg→rad → clip → EMA.
        """
        # Step 1 — per-finger 4-vectors in [Spread, MCP, PIP, DIP] order (deg).
        # index_vals[0] is always 0 because the Manus publisher drops
        # IndexMCPSpread (ErgonomicsDataTypeToSide bug).
        index_vals  = np.array(glove16[0:4],   dtype=float)
        middle_vals = np.array(glove16[4:8],   dtype=float)
        ring_vals   = np.array(glove16[8:12],  dtype=float)
        thumb_vals  = np.array(glove16[12:16], dtype=float)

        # Step 2 — raw `scale × ergo_deg`. Joint range annotations below are
        # from ALLEGRO_LOWER/UPPER_LIMITS (converted to deg for readability)
        # and explain WHY each scale sign was chosen.
        s = self.scales
        angle_deg = np.array([
            # ── Index (joint_0..3) ────────────────────────────────────
            # j0 [±27°]: spread. IndexMCPSpread is dropped by the
            # publisher (val[0] always 0), so joint_0 is synthesized
            # here from the SAME middle (m20) and ring (r30) raw scale
            # terms used below, mixed via tunable weights
            # `i_from_m` / `i_from_r` (default 0.5 / 0.5).
            (s['i_from_m'] * s['m20'] * middle_vals[0]
             + s['i_from_r'] * s['r30'] * ring_vals[0]),
            s['i11'] * index_vals[1],   # j1 [-11..+92°]: MCP flex (+ → +)
            s['i12'] * index_vals[2],   # j2 [-10..+98°]: PIP flex (+ → +)
            s['i13'] * index_vals[3],   # j3 [-13..+93°]: DIP flex (+ → +)
            # ── Middle (joint_4..7) ───────────────────────────────────
            # j4 [±27°]: spread. m20 < 0 (empirical sign chosen so the
            # right-hand glove's middle-spread polarity aligns with
            # Allegro's joint-4 axis convention).
            s['m20'] * middle_vals[0],
            s['m21'] * middle_vals[1],  # j5 [-11..+92°]: MCP flex
            s['m22'] * middle_vals[2],  # j6 [-10..+98°]: PIP flex
            s['m23'] * middle_vals[3],  # j7 [-13..+93°]: DIP flex
            # ── Ring (joint_8..11) ────────────────────────────────────
            # j8 [±27°]: spread. r30 > 0: direct mapping consistent with
            # the right-hand convention that ring splay (ulnar) shows
            # up as negative Manus Spread → negative Allegro joint.
            s['r30'] * ring_vals[0],
            s['r31'] * ring_vals[1],    # j9  [-11..+92°]: MCP flex
            s['r32'] * ring_vals[2],    # j10 [-10..+98°]: PIP flex
            s['r33'] * ring_vals[3],    # j11 [-13..+93°]: DIP flex
            # ── Thumb (joint_12..15) — cross-channel v[0]↔v[1] swap ──
            # j12 [+40..+80°]: CMC-like, entirely POSITIVE range. Driven
            # by val[1] (ThumbMCPStretch, CMC flex) with NEGATIVE scale
            # (-4.375) and LARGE offset (o_t00 = 225° ≈ 90°×2.5 legacy).
            # At rest (val[1]=0): angle = 225°; clips to upper 80°.
            # More CMC flex → joint decreases toward lower 40°.
            s['t00'] * thumb_vals[1],
            # j13 [+17..+67°]: base swing, entirely POSITIVE range.
            # Driven by val[0] (ThumbMCPSpread, palmar abduction) with
            # positive scale 6.0 and offset 0°. At rest val[0]=0 → 0° →
            # clips to lower 17°; more abduction lifts the joint.
            s['t01'] * thumb_vals[0],
            s['t02'] * thumb_vals[2],   # j14 [-11..+94°]: MCP flex (o_t02 = -30°)
            s['t03'] * thumb_vals[3],   # j15 [-9..+98°]:  IP  flex
        ], dtype=float)

        # Step 3 — additive per-joint offsets (deg). Sliders mutate
        # self.offsets live. Thumb offsets are LARGE (o_t00 = 225°, etc.)
        # because Allegro thumb joints have positive-biased ranges that
        # don't include 0° — the offsets shift the rest pose into the
        # valid joint range.
        o = self.offsets
        offset_deg = np.array([
            o['o_i10'], o['o_i11'], o['o_i12'], o['o_i13'],
            o['o_m20'], o['o_m21'], o['o_m22'], o['o_m23'],
            o['o_r30'], o['o_r31'], o['o_r32'], o['o_r33'],
            o['o_t00'], o['o_t01'], o['o_t02'], o['o_t03'],
        ], dtype=float)
        angle_deg = angle_deg + offset_deg

        # Steps 4-5 — deg → rad, clip to Allegro limits. The clip also
        # discards glove-side hyperextension (Allegro flex joints stop
        # around −11°..−13°), per ERGONOMICS.md §4 item 3.
        arr = np.deg2rad(angle_deg)
        arr = np.clip(arr, ALLEGRO_LOWER_LIMITS, ALLEGRO_UPPER_LIMITS)

        # Step 6 — EMA smoothing.
        if self.prev_arr is None:
            smoothed = arr.copy()
        else:
            smoothed = self.alpha * arr + (1.0 - self.alpha) * self.prev_arr
        self.prev_arr = smoothed
        return smoothed

    def _ergonomics_to_array(self, ergonomics_list):
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
        self.latest_qpos = self.transform_glove_to_allegro(glove16)

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
        description='Allegro hand rule-based retargeting (simulator verification).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python allegro_rule_based_retargeting.py
  python allegro_rule_based_retargeting.py --no-kinematic
  python allegro_rule_based_retargeting.py --no-tune
  python allegro_rule_based_retargeting.py --hand allegro_left --side left
        """
    )
    parser.add_argument('--hand', type=str, default='allegro_right')
    parser.add_argument('--side', type=str, choices=['left', 'right'], default='right')
    parser.add_argument('--no-kinematic', action='store_false', dest='kinematic',
                        help='Physics-mode PD drive (default is kinematic snap).')
    parser.add_argument('--no-ergo-viz', action='store_true',
                        help='Disable the Open3D ergonomics window.')
    parser.add_argument('--no-tune', action='store_false', dest='tune',
                        help='Disable the tkinter live tuning slider window.')
    args = parser.parse_args()

    # Sapien
    engine, renderer, scene = ru.setup_sapien_scene()
    config = get_config(args.hand)
    print(f"[allegro_rule_based_retargeting] loading config: {config['name']}")

    model = HandKinematicModel.build_from_config(config, scene=scene, render=False)
    model.hand.set_root_pose(sapien.Pose([0.0, 0.0, 0.35], [0.695, 0.0, -0.718, 0.0]))
    ru.colorize_articulation(model.hand, _allegro_link_classifier)

    if not args.kinematic:
        ru.tune_pd_drives(model)

    links = [info['link'] for info in config['fingertip_link']]
    offsets = [info['center_offset'] for info in config['fingertip_link']]
    model.initialize_keypoint(links, offsets)
    viewer_env = HandViewerEnv([model], scene=scene, renderer=renderer)

    ergo_viz = None
    if not args.no_ergo_viz:
        ergo_viz = ru.ErgonomicsBarViz(
            ALLEGRO_LOWER_LIMITS, ALLEGRO_UPPER_LIMITS,
            fingers=['index', 'middle', 'ring', 'thumb'],
            finger_rgb=ALLEGRO_FINGER_RGB,
            window_title='Manus ergonomics (front) / Allegro joint fraction (back)',
            window_size=(1000, 620),
        )

    # ROS
    rclpy.init()
    node = ManusAllegroSimNode(side=args.side)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    tuner = None
    if args.tune:
        try:
            tuner = ru.TuningSliders(
                node,
                scale_layout=ALLEGRO_SCALE_LAYOUT,
                offset_layout=ALLEGRO_OFFSET_LAYOUT,
                scale_magnitude_max=7.0, scale_res=0.1,
                offset_min=-50.0, offset_max=250.0, offset_res=1.0,
                slider_length=220,
                with_load=True, with_record=True, default_record_hz=50,
                with_broadcast=False,
                title='Allegro Retargeting Tuning',
            )
            print("[allegro_rule_based_retargeting] tuning slider window open.")
        except RuntimeError as e:
            print(f"[allegro_rule_based_retargeting] failed to open tuner: {e}")

    mode_str = 'kinematic snap' if args.kinematic else 'physics / PD'
    print(f"[allegro_rule_based_retargeting] running in {mode_str} mode. "
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
