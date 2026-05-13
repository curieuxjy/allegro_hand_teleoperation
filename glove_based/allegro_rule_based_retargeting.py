#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
Manus Glove to Allegro Hand Rule-Based Retargeting — Simulator Verification

Mirrors v6_rule_based_retargeting.py but for the Allegro 16-DOF hand
(4 fingers x 4 joints, no pinky). The defaults come from the empirical
mapping in rule_based_retargeting.py (the hardware-publishing variant),
re-expressed in the scale + offset pipeline so the same live tuning
slider window can edit them.

Architecture
  - ROS2 spin runs in a background thread (subscribes to /manus_glove_{side}).
  - Glove callback parses ergonomics, computes a 16-D Allegro qpos in
    `self.latest_qpos`, optionally appends a frame to the recording buffer.
  - Main thread runs the Sapien viewer loop; each frame it reads the latest
    qpos and applies it to the articulation (kinematic snap or PD target).

CLI
  --hand          : config name (default allegro_right)
  --side          : glove side to subscribe to (default right)
  --no-kinematic  : opt out of kinematic snap (default ON)
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
import open3d as o3d
import rclpy
from rclpy.node import Node
import sapien.core as sapien
from manus_ros2_msgs.msg import ManusGlove

# Path setup for direct script execution
try:
    from .geort.utils.config_utils import get_config
    from .geort.env.hand_debug import HandKinematicModel, HandViewerEnv
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from geort.utils.config_utils import get_config
    from geort.env.hand_debug import HandKinematicModel, HandViewerEnv


# Joint limits for Allegro right hand (radians), in the order the sim expects:
# allegro_right.json's `joint_order` is [joint_0..3, joint_4..7, joint_8..11,
# joint_12..15] = Index → Middle → Ring → Thumb. The legacy hardware-publishing
# script used thumb-first ordering; for sim we follow the config.
ALLEGRO_UPPER_LIMITS = np.array([
     0.4700,  1.6100,  1.7090,  1.6180,   # index  (joint_0..3)
     0.4700,  1.6100,  1.7090,  1.6180,   # middle (joint_4..7)
     0.4700,  1.6100,  1.7090,  1.6180,   # ring   (joint_8..11)
     1.3960,  1.1630,  1.6440,  1.7190,   # thumb  (joint_12..15)
])

ALLEGRO_LOWER_LIMITS = np.array([
    -0.4700, -0.1960, -0.1740, -0.2270,   # index
    -0.4700, -0.1960, -0.1740, -0.2270,   # middle
    -0.4700, -0.1960, -0.1740, -0.2270,   # ring
     0.7000,  0.3000, -0.1890, -0.1620,   # thumb
])


class ManusAllegroSimNode(Node):
    """
    ROS2 subscriber: turns Manus glove messages into Allegro joint angles.

    Stores the latest 16-D qpos (user order, radians) in `self.latest_qpos`
    for the main thread to consume. Tunable constants live in `self.scales`
    and `self.offsets` and can be mutated live from TuningSliders.

    Allegro thumb convention (from the legacy rule_based_retargeting.py):
      - joint_12 (CMC)          driven by glove val[1]   (NOT val[0])
      - joint_13 (base swing)   driven by glove val[0]   (NOT val[1])
      - joint_14 (MCP-like)     driven by glove val[2]
      - joint_15 (IP-like)      driven by glove val[3]
    The val[0]<->val[1] swap is intentional — Manus thumb spread/CMC sensors
    map to Allegro thumb's CMC/base axes in this inverted way.
    """

    # Step 2 default scales. Thumb coefficients fold the legacy post-rad
    # multipliers (`*2.5`, `*2`) into a single pre-deg2rad scale.
    DEFAULT_SCALES = {
        # Thumb
        't00': -4.375,   # arr[0] (CMC, joint_12)  ← (-1.75 * v[1]) * 2.5
        't01':  6.0,     # arr[1] (base, joint_13) ← (3.0 * v[0]) * 2
        't02':  3.0,     # arr[2] (mid,  joint_14) ← 3.0 * v[2]
        't03':  2.0,     # arr[3] (tip,  joint_15) ← v[3] * 2
        # Index
        'i10': -0.5,
        'i11':  1.5,
        'i12':  1.0,
        'i13':  2.0,
        # Middle
        'm20': -0.2,
        'm21':  1.5,
        'm22':  1.0,
        'm23':  2.0,
        # Ring
        'r30':  0.1,
        'r31':  1.5,
        'r32':  1.0,
        'r33':  2.0,
    }

    # Step 3 default offsets (degrees). Thumb offsets absorb the legacy
    # baselines (90, -45, -30, etc.) plus the post-rad scaling.
    DEFAULT_OFFSETS = {
        'o_t00':  225.0,   # = 90 * 2.5
        'o_t01':    0.0,   # = -45 * 2 + 90  →  0
        'o_t02':  -30.0,
        'o_t03':    0.0,
        'o_i10':    0.0, 'o_i11': 0.0, 'o_i12': 0.0, 'o_i13': 0.0,
        'o_m20':   -4.0,   # = 20 * (-0.2)
        'o_m21':    0.0, 'o_m22': 0.0, 'o_m23': 0.0,
        'o_r30':    0.0, 'o_r31': 0.0, 'o_r32': 0.0,
        'o_r33':   10.0,   # = 5 * 2
    }

    TUNING_FILE = Path(__file__).resolve().parent / 'allegro_tuning.json'
    RECORDING_DIR = Path(__file__).resolve().parent / 'recordings'

    # Motor encoder conversion: 12-bit encoder, one revolution = 4096 counts.
    DEG_TO_COUNTS = 4096.0 / 360.0

    def __init__(self, side: str = 'right'):
        super().__init__('manus_allegro_sim_node')
        self.side = side.lower()

        # EMA smoothing coefficient
        self.alpha = 0.2
        self.prev_arr = None

        # Live-tunable constants
        self.scales = dict(self.DEFAULT_SCALES)
        self.offsets = dict(self.DEFAULT_OFFSETS)
        self._try_load_tuning()

        # Shared state for main thread — both 16-D in sim joint order:
        # [Index 0..3, Middle 4..7, Ring 8..11, Thumb 12..15]
        self.latest_qpos = None      # 16-D Allegro joint angles (rad)
        self.latest_glove16 = None   # 16-D raw glove ergonomics (deg)
        self._got_first = False

        # Recording state
        self.recording = False
        self.recorded_frames = []
        self._rec_t0 = None
        self.last_record_stats = None

        topic = f'/manus_glove_{self.side}'
        self.create_subscription(ManusGlove, topic, self.glove_callback, 20)
        self.get_logger().info(f'Subscribed to {topic}')

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def start_recording(self):
        self.recorded_frames = []
        self._rec_t0 = time.monotonic()
        self.recording = True
        self.get_logger().info('Motion recording started.')

    def stop_recording(self):
        """Stop recording and write frames as encoder counts to a timestamped .txt.

        Format: frames concatenated as `[v0, v1, ..., v15][v0, v1, ..., v15]...`
        with 16 integer values per frame in URDF order
        (thumb → index → middle → ring), each value
        = round(rad → deg × (4096 / 360)).
        Returns the saved Path on success, or None if nothing was recorded.
        """
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

    # ------------------------------------------------------------------
    # Tuning load/save
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Glove → Allegro retargeting
    # ------------------------------------------------------------------
    def transform_glove_to_allegro(self, glove16):
        """
        Transform 16-D glove ergonomics → 16-D Allegro joint angles.
        Both arrays use the sim joint order:
          [Index 0..3, Middle 4..7, Ring 8..11, Thumb 12..15],
        each finger's 4 entries being [spread, MCP, PIP, DIP] glove values.

        Pipeline:
          1. Extract per-finger glove values.
          2. Per-joint scale (degrees).
          3. Per-joint offset (degrees).
          4. Convert degrees → radians.
          5. Clip to Allegro joint limits.
          6. EMA smoothing.
        """
        # Step 1 — glove16 layout (post-parse) matches sim joint order:
        #   [0..3] index, [4..7] middle, [8..11] ring, [12..15] thumb.
        index_vals  = np.array(glove16[0:4],   dtype=float)
        middle_vals = np.array(glove16[4:8],   dtype=float)
        ring_vals   = np.array(glove16[8:12],  dtype=float)
        thumb_vals  = np.array(glove16[12:16], dtype=float)

        # Step 2: per-joint scale (deg). arr order = sim joint_order =
        # [index 0..3, middle 4..7, ring 8..11, thumb 12..15].
        #
        # NOTE on thumb: the legacy mapping cross-wires val[0] and val[1] —
        #   joint_12 (CMC) is driven by val[1] (Manus "MCP stretch") while
        #   joint_13 (base swing) is driven by val[0] (Manus "MCP spread").
        # This is preserved here.
        s = self.scales
        angle_deg = np.array([
            # Index (joint_0..3)
            s['i10'] * index_vals[0],
            s['i11'] * index_vals[1],
            s['i12'] * index_vals[2],
            s['i13'] * index_vals[3],
            # Middle (joint_4..7)
            s['m20'] * middle_vals[0],
            s['m21'] * middle_vals[1],
            s['m22'] * middle_vals[2],
            s['m23'] * middle_vals[3],
            # Ring (joint_8..11)
            s['r30'] * ring_vals[0],
            s['r31'] * ring_vals[1],
            s['r32'] * ring_vals[2],
            s['r33'] * ring_vals[3],
            # Thumb (joint_12..15)
            s['t00'] * thumb_vals[1],   # CMC  ← v[1]   (cross channel)
            s['t01'] * thumb_vals[0],   # base ← v[0]  (cross channel)
            s['t02'] * thumb_vals[2],   # mid
            s['t03'] * thumb_vals[3],   # tip
        ], dtype=float)

        # Step 3: per-joint offset (deg) — same order as Step 2.
        o = self.offsets
        offset_deg = np.array([
            o['o_i10'], o['o_i11'], o['o_i12'], o['o_i13'],
            o['o_m20'], o['o_m21'], o['o_m22'], o['o_m23'],
            o['o_r30'], o['o_r31'], o['o_r32'], o['o_r33'],
            o['o_t00'], o['o_t01'], o['o_t02'], o['o_t03'],
        ], dtype=float)
        angle_deg = angle_deg + offset_deg

        # Step 4: deg → rad
        arr = np.deg2rad(angle_deg)

        # Step 5: clip
        arr = np.clip(arr, ALLEGRO_LOWER_LIMITS, ALLEGRO_UPPER_LIMITS)

        # Step 6: EMA
        if self.prev_arr is None:
            smoothed = arr.copy()
        else:
            smoothed = self.alpha * arr + (1.0 - self.alpha) * self.prev_arr
        self.prev_arr = smoothed
        return smoothed

    # ------------------------------------------------------------------
    # Ergonomics parsing — same publisher quirks as v6.
    # Output layout matches the sim's joint order so glove16[i] feeds the
    # glove channel that drives arr[i] in transform_glove_to_allegro.
    # ------------------------------------------------------------------
    _FINGER_OFFSET = {'Index': 0, 'Middle': 4, 'Ring': 8, 'Thumb': 12}
    _MOTION_IDX = {
        'MCPSpread':  0,
        'Spread':     0,   # publisher uses "MiddleSpread" etc. for non-thumb fingers
        'MCPStretch': 1,
        'PIPStretch': 2,
        'DIPStretch': 3,
    }

    def _ergonomics_to_array(self, ergonomics_list):
        """Parse Manus ergonomics list into a 16-D array in sim joint order
        (index → middle → ring → thumb). Pinky values are ignored.
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
        self.latest_qpos = self.transform_glove_to_allegro(glove16)

        if self.recording:
            self.recorded_frames.append(self.latest_qpos.copy())

        if not self._got_first:
            self._got_first = True
            self.get_logger().info(
                f'First glove command received ({len(msg.ergonomics)} entries); driving sim.'
            )


# ----------------------------------------------------------------------
# Sapien helpers
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


def _tune_allegro_drives(model: HandKinematicModel,
                         kp: float = 400.0, kd: float = 30.0,
                         force_limit: float = 10.0):
    """Higher kd than HandKinematicModel's default for self-collision stability."""
    for j in model.all_joints:
        j.set_drive_property(kp, kd, force_limit=force_limit)


# Per-finger colors (no pinky entry — Allegro has 4 fingers).
FINGER_COLORS = {
    'link_12': [0.95, 0.30, 0.30, 1.0],   # thumb (red) — joints 12..15
    'link_13': [0.95, 0.30, 0.30, 1.0],
    'link_14': [0.95, 0.30, 0.30, 1.0],
    'link_15': [0.95, 0.30, 0.30, 1.0],
    'link_0':  [0.30, 0.55, 0.95, 1.0],   # index (blue) — joints 0..3
    'link_1':  [0.30, 0.55, 0.95, 1.0],
    'link_2':  [0.30, 0.55, 0.95, 1.0],
    'link_3':  [0.30, 0.55, 0.95, 1.0],
    'link_4':  [0.30, 0.85, 0.40, 1.0],   # middle (green) — joints 4..7
    'link_5':  [0.30, 0.85, 0.40, 1.0],
    'link_6':  [0.30, 0.85, 0.40, 1.0],
    'link_7':  [0.30, 0.85, 0.40, 1.0],
    'link_8':  [0.95, 0.65, 0.20, 1.0],   # ring (orange) — joints 8..11
    'link_9':  [0.95, 0.65, 0.20, 1.0],
    'link_10': [0.95, 0.65, 0.20, 1.0],
    'link_11': [0.95, 0.65, 0.20, 1.0],
}

# Finger → RGB for the ergonomics bars (matches the link colors above).
ERGO_FINGER_RGB = {
    'thumb':  [0.95, 0.30, 0.30],
    'index':  [0.30, 0.55, 0.95],
    'middle': [0.30, 0.85, 0.40],
    'ring':   [0.95, 0.65, 0.20],
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


def _colorize_allegro_fingers(articulation):
    """Color Allegro links by finger. Allegro URDF uses `link_N.0` naming
    (e.g. `link_0.0`, `link_12.0`), so we strip the trailing `.0` and match."""
    for link in articulation.get_links():
        name = link.get_name()
        # Strip Allegro suffix variants: 'link_0.0', 'link_0.0_tip', etc.
        base = name.split('.')[0] if name.startswith('link_') else name
        rgba = FINGER_COLORS.get(base)
        if rgba is not None:
            _set_link_color(link, rgba)


# ----------------------------------------------------------------------
# Open3D ergonomics + joint-fraction debug window (16 channels)
# ----------------------------------------------------------------------
class ErgonomicsBarViz:
    """Open3D dual-row bar chart for the Allegro mapping (16 channels).

    Front row: raw Manus ergonomics (deg, finger order thumb→ring × [spread,
    MCP, PIP, DIP]).
    Back row: 16-D Allegro joint angle as fraction of [lower, upper] range.
    A red line on the back row marks fraction = 1.0 (joint upper limit).
    """

    FINGERS = ['index', 'middle', 'ring', 'thumb']
    MOTIONS = ['spread', 'mcp', 'pip', 'dip']
    N = 16

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
            window_name='Manus ergonomics (front) / Allegro joint fraction (back)',
            width=1000, height=620,
        )

        self.boxes_top, self.init_verts_top = [], []
        self.boxes_bot, self.init_verts_bot = [], []
        self.bar_x = np.zeros(self.N)

        z_top = -row_z_gap / 2
        z_bot = +row_z_gap / 2

        for i in range(self.N):
            f_idx, m_idx = i // 4, i % 4
            finger = self.FINGERS[f_idx]
            x = f_idx * finger_spacing + m_idx * motion_spacing
            self.bar_x[i] = x

            base = ERGO_FINGER_RGB[finger]
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

        # Reference geometry
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(frame)

        x_min = -0.5
        x_max = 3 * finger_spacing + 3 * motion_spacing + 0.5
        ref = o3d.geometry.LineSet()
        ref.points = o3d.utility.Vector3dVector([
            [x_min, 0.0,        z_top],
            [x_max, 0.0,        z_top],
            [x_min, 0.0,        z_bot],
            [x_max, 0.0,        z_bot],
            [x_min, frac_scale, z_bot],
            [x_max, frac_scale, z_bot],
        ])
        ref.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3], [4, 5]])
        ref.colors = o3d.utility.Vector3dVector([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.85, 0.25, 0.25],
        ])
        self.vis.add_geometry(ref)

        self.update(np.zeros(self.N), np.zeros(self.N))
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

    def update(self, glove16, qpos16):
        if glove16 is not None and len(glove16) >= self.N:
            for i in range(self.N):
                h = float(glove16[i]) * self.deg_scale
                self._resize_bar(self.boxes_top[i], self.init_verts_top[i], h)
        if qpos16 is not None and len(qpos16) >= self.N:
            q = np.asarray(qpos16, dtype=float)
            normalized = (q - self.joint_lower) / self.joint_range
            for i in range(self.N):
                h = float(normalized[i]) * self.frac_scale
                self._resize_bar(self.boxes_bot[i], self.init_verts_bot[i], h)

    def render(self):
        if self._first_render:
            ctr = self.vis.get_view_control()
            ctr.set_lookat([5.5, 0.8, 0.0])
            ctr.set_front([0.0, -0.3, 1.0])
            ctr.set_up([0.0, 1.0, 0.0])
            ctr.set_zoom(0.55)
            self._first_render = False
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass

    @staticmethod
    def print_legend():
        print("[ergo viz] Two rows along Z: front = ergonomics (deg, scaled), "
              "back = joint angle fraction (0..1 of [lower, upper]).")
        print("[ergo viz] Back-row red line = fraction 1.0 (joint upper limit).")
        print("[ergo viz] X order: index → middle → ring → thumb (sim joint order), "
              "each [spread, MCP, PIP, DIP].")
        print("[ergo viz] Color: index=blue, middle=green, ring=orange, thumb=red.")


# ----------------------------------------------------------------------
# Tkinter slider window — 4 finger columns, single-channel sliders
# ----------------------------------------------------------------------
class TuningSliders:
    """Tkinter slider window for live editing Allegro Step 2 scales and
    Step 3 offsets. Sign on scale sliders is locked to the default's sign."""

    # Column order matches the sim joint order (Index, Middle, Ring, Thumb).
    _SCALE_LAYOUT = [
        ('Index', [
            ('i10', '10 (spread)'),
            ('i11', '11 (MCP)'),
            ('i12', '12 (PIP)'),
            ('i13', '13 (DIP)'),
        ]),
        ('Middle', [
            ('m20', '20 (spread)'),
            ('m21', '21 (MCP)'),
            ('m22', '22 (PIP)'),
            ('m23', '23 (DIP)'),
        ]),
        ('Ring', [
            ('r30', '30 (spread)'),
            ('r31', '31 (MCP)'),
            ('r32', '32 (PIP)'),
            ('r33', '33 (DIP)'),
        ]),
        ('Thumb', [
            ('t00', '00 CMC ← v[1]'),
            ('t01', '01 base ← v[0]'),
            ('t02', '02 mid ← v[2]'),
            ('t03', '03 tip ← v[3]'),
        ]),
    ]
    _OFFSET_LAYOUT = [
        ('Index',  ['o_i10', 'o_i11', 'o_i12', 'o_i13']),
        ('Middle', ['o_m20', 'o_m21', 'o_m22', 'o_m23']),
        ('Ring',   ['o_r30', 'o_r31', 'o_r32', 'o_r33']),
        ('Thumb',  ['o_t00', 'o_t01', 'o_t02', 'o_t03']),
    ]

    _FONT_HEADER  = ('TkDefaultFont', 13, 'bold')
    _FONT_SECTION = ('TkDefaultFont', 11, 'italic')
    _FONT_LABEL   = ('TkDefaultFont', 11)
    _FONT_SLIDER  = ('TkDefaultFont', 10)
    _FONT_BUTTON  = ('TkDefaultFont', 11, 'bold')

    def __init__(self, node,
                 scale_magnitude_max=7.0, scale_res=0.1,
                 offset_min=-50.0, offset_max=250.0, offset_res=1.0,
                 slider_length=220):
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
        self.root.title('Allegro Retargeting Tuning')

        self._slider_vars = {}

        outer = tk.Frame(self.root)
        outer.pack(padx=8, pady=8)

        for col_idx, (finger, scale_pairs) in enumerate(self._SCALE_LAYOUT):
            col = tk.LabelFrame(outer, text=finger,
                                font=self._FONT_HEADER,
                                padx=6, pady=6)
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

        # Bottom action row 1: Save + Reset
        btn_row = tk.Frame(self.root)
        btn_row.pack(pady=(0, 4))
        tk.Button(btn_row, text='Save', font=self._FONT_BUTTON,
                  width=12, command=self._on_save).pack(side='left', padx=4)
        tk.Button(btn_row, text='Reset to defaults', font=self._FONT_BUTTON,
                  width=18, command=self._reset_to_defaults).pack(side='left', padx=4)

        # Bottom action row 2: Recording controls
        rec_row = tk.Frame(self.root)
        rec_row.pack(pady=(0, 4))
        self._rec_button = tk.Button(
            rec_row, text='● Start Record', font=self._FONT_BUTTON,
            width=18, fg='#a00', command=self._on_record_toggle,
        )
        self._rec_button.pack(side='left', padx=4)
        self._rec_status_var = tk.StringVar(value='')
        tk.Label(rec_row, textvariable=self._rec_status_var,
                 font=self._FONT_LABEL, fg='#a00').pack(side='left', padx=4)

        self._status_var = tk.StringVar(value='')
        tk.Label(self.root, textvariable=self._status_var,
                 font=self._FONT_LABEL, fg='#0a0').pack(pady=(0, 4))

        self._tick_rec_status()

    def _make_scale_slider(self, parent, key, label):
        default = self.node.scales[key]
        mag = self.scale_magnitude_max
        if default < 0:
            mn, mx = -mag, 0.0
        else:
            mn, mx = 0.0, mag
        self._make_slider(parent, self.node.scales, key, label,
                          mn, mx, self.scale_res)

    def _make_offset_slider(self, parent, key, label):
        self._make_slider(parent, self.node.offsets, key, label,
                          self.offset_min, self.offset_max, self.offset_res)

    def _make_slider(self, parent, store, key, label, mn, mx, res):
        tk = self.tk
        row = tk.Frame(parent)
        row.pack(fill='x', pady=2)
        tk.Label(row, text=label, width=14, anchor='w',
                 font=self._FONT_LABEL).pack(side='left')

        var = tk.DoubleVar(value=store[key])

        def on_change(value, k=key, s=store):
            try:
                s[k] = float(value)
            except (TypeError, ValueError):
                pass

        slider = tk.Scale(row, from_=mn, to=mx, resolution=res,
                          orient='horizontal', variable=var,
                          length=self.slider_length,
                          showvalue=True, command=on_change,
                          font=self._FONT_SLIDER)
        slider.pack(side='left')
        self._slider_vars[(id(store), key)] = (var, store)

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
            self.node.start_recording()
            self._rec_button.config(text='■ Stop Record', fg='#080')

    def _tick_rec_status(self):
        if self.node.recording:
            n = len(self.node.recorded_frames)
            self._rec_status_var.set(f'REC  {n} frames')
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
        description='Allegro hand rule-based retargeting (simulator verification).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: kinematic snap + live tuning sliders, /manus_glove_right
  python allegro_rule_based_retargeting.py

  # Physics-mode PD drive instead of kinematic snap
  python allegro_rule_based_retargeting.py --no-kinematic

  # Skip the slider window
  python allegro_rule_based_retargeting.py --no-tune

  # Left glove
  python allegro_rule_based_retargeting.py --hand allegro_left --side left
        """
    )
    parser.add_argument('--hand', type=str, default='allegro_right',
                        help='Config name to load (default: allegro_right)')
    parser.add_argument('--side', type=str, choices=['left', 'right'], default='right',
                        help='Glove side to subscribe to (default: right)')
    parser.add_argument('--no-kinematic', action='store_false', dest='kinematic',
                        help='Run physics-mode PD drive (default is kinematic snap).')
    parser.add_argument('--no-ergo-viz', action='store_true',
                        help='Disable the Open3D ergonomics / joint-fraction window.')
    parser.add_argument('--no-tune', action='store_false', dest='tune',
                        help='Disable the live tuning slider window.')
    args = parser.parse_args()

    # Sapien
    engine, renderer, scene = _setup_sapien_scene()

    config = get_config(args.hand)
    print(f"[allegro_rule_based_retargeting] loading config: {config['name']}")

    model = HandKinematicModel.build_from_config(config, scene=scene, render=False)
    model.hand.set_root_pose(sapien.Pose([0.0, 0.0, 0.35], [0.695, 0.0, -0.718, 0.0]))

    _colorize_allegro_fingers(model.hand)

    if not args.kinematic:
        _tune_allegro_drives(model)

    links = [info['link'] for info in config['fingertip_link']]
    offsets = [info['center_offset'] for info in config['fingertip_link']]
    model.initialize_keypoint(links, offsets)

    viewer_env = HandViewerEnv([model], scene=scene, renderer=renderer)

    # Ergo viz
    ergo_viz = None
    if not args.no_ergo_viz:
        ergo_viz = ErgonomicsBarViz(ALLEGRO_LOWER_LIMITS, ALLEGRO_UPPER_LIMITS)
        ErgonomicsBarViz.print_legend()

    # ROS
    rclpy.init()
    node = ManusAllegroSimNode(side=args.side)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Tuner
    tuner = None
    if args.tune:
        try:
            tuner = TuningSliders(node)
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
                _apply_qpos(model, qpos, args.kinematic)

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
