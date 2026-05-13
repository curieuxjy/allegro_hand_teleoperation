#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
Manus Glove to V6 Hand Rule-Based Retargeting — Simulator Verification

Like rule_based_retargeting.py, but instead of publishing to a hardware
controller, this script drives the V6 hand inside a Sapien simulator
(via HandKinematicModel + HandViewerEnv, see geort/env/hand_debug.py).
Intended use: visually verify that the heuristic glove→v6 mapping produces
sensible joint angles before connecting to real hardware.

Architecture
  - ROS2 spin runs in a background thread (subscribes to /manus_glove_{side}).
  - Glove callback computes the 20-D V6 qpos (user order) and stashes it.
  - Main thread runs the Sapien viewer loop; each frame it reads the latest
    qpos and applies it to the articulation (kinematic snap or PD target).

CLI
  --hand          : config name (default v6_right)
  --side          : glove side to subscribe to (default right)
  --no-kinematic  : opt out of kinematic snap mode (default is ON; physics mode
                    is available but may wobble on v6's 5-finger / 20-DOF chain
                    under PD + self-collision; see hand_debug.py docstring).
  --no-tune       : opt out of the live tuning slider window (default is ON).
  --no-ergo-viz   : opt out of the Open3D ergonomics / joint-fraction debug window.


  # 기본: kinematic + tune 둘 다 켜진 상태
  python glove_based/v6_rule_based_retargeting.py

  # Physics 모드로 돌리고 싶을 때
  python glove_based/v6_rule_based_retargeting.py --no-kinematic

  # 슬라이더 창 빼고
  python glove_based/v6_rule_based_retargeting.py --no-tune

  # 둘 다 빼고 ergo viz도 빼면 가장 가벼움
  python glove_based/v6_rule_based_retargeting.py --no-tune --no-ergo-viz

"""

import sys
import argparse
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
import sapien.core as sapien
from manus_ros2_msgs.msg import ManusGlove

# Path setup for direct script execution (mirrors rule_based_retargeting.py)
try:
    from .geort.utils.config_utils import get_config
    from .geort.env.hand_debug import HandKinematicModel, HandViewerEnv
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from geort.utils.config_utils import get_config
    from geort.env.hand_debug import HandKinematicModel, HandViewerEnv


# Joint limits for V6 right hand (radians), in URDF/config user order:
#   Thumb (joint_00..03) -> Index (joint_10..13) -> Middle (joint_20..23) ->
#   Ring (joint_30..33) -> Pinky (joint_40..43).
# These match v6_right.urdf and v6_right.json's `joint_order`.
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


class ManusV6SimNode(Node):
    """
    ROS2 subscriber: turns Manus glove messages into V6 joint angles.

    Stores the latest 20-D qpos (user order, radians) in `self.latest_qpos`
    for the main thread to consume.

    Tunable constants live in `self.scales` (Step 2 multipliers) and
    `self.offsets` (Step 3 additive biases, in degrees). The dicts seed
    from `DEFAULT_SCALES` / `DEFAULT_OFFSETS` and can be mutated live
    (e.g. by TuningSliders) — dict reads/writes are atomic in CPython
    so no lock is needed between the ROS callback and the slider thread.
    """

    # Step 2 default scales — joint_00 and joint_40 use TWO entries each
    # because curl and spread are mixed (see commentary inline in transform).
    DEFAULT_SCALES = {
        # Thumb
        't00_curl':   -0.5,
        't00_spread': -1.5,
        't01':         1.5,
        't02':         1.0,
        't03':         1.0,
        # Index
        'i10':         1.0,
        'i11':         1.5,
        'i12':         1.0,
        'i13':         2.0,
        # Middle
        'm20':         0.3,
        'm21':         1.5,
        'm22':         1.0,
        'm23':         2.0,
        # Ring
        'r30':        -1.8,
        'r31':         1.5,
        'r32':         2.0,
        'r33':         1.5,
        # Pinky
        'p40_curl':   -0.5,
        'p40_spread': -1.5,
        'p41':        -0.9,
        'p42':         2.5,
        'p43':         2.0,
    }

    # Step 3 default offsets (degrees)
    DEFAULT_OFFSETS = {
        'o_t00':  0.0, 'o_t01': 0.0, 'o_t02': 0.0, 'o_t03': -3.0,
        'o_i10':  0.0, 'o_i11': 0.0, 'o_i12': 0.0, 'o_i13': -3.0,
        'o_m20':  0.0, 'o_m21': 0.0, 'o_m22': 0.0, 'o_m23': -3.0,
        'o_r30': -3.0, 'o_r31': 0.0, 'o_r32': 0.0, 'o_r33': -3.0,
        'o_p40':  0.0, 'o_p41': -10.0, 'o_p42': -10.0, 'o_p43': -5.0,
    }

    TUNING_FILE = Path(__file__).resolve().parent / 'v6_tuning.json'
    RECORDING_DIR = Path(__file__).resolve().parent / 'recordings'

    # Motor encoder conversion: 12-bit encoder, one revolution = 4096 counts.
    DEG_TO_COUNTS = 4096.0 / 360.0

    def __init__(self, side: str = 'right'):
        super().__init__('manus_v6_sim_node')
        self.side = side.lower()

        # EMA smoothing coefficient (0 < alpha <= 1, higher = more current measurement)
        self.alpha = 0.2
        self.prev_arr = None

        # Live-tunable constants (start from defaults; sliders can mutate)
        self.scales = dict(self.DEFAULT_SCALES)
        self.offsets = dict(self.DEFAULT_OFFSETS)
        # If the user has saved a tuning previously, restore it on startup.
        self._try_load_tuning()

        # Shared state for the Sapien main thread to read
        self.latest_qpos = None      # 20-D V6 joint angles (rad, user order)
        self.latest_glove20 = None   # 20-D raw glove ergonomics (deg)
        self._got_first = False

        # Recording state (toggled from the tuner window)
        self.recording = False
        self.recorded_frames = []    # list of np.ndarray (20-D, radians)

        topic = f'/manus_glove_{self.side}'
        self.create_subscription(ManusGlove, topic, self.glove_callback, 20)
        self.get_logger().info(f'Subscribed to {topic}')

    def start_recording(self):
        self.recorded_frames = []
        self.recording = True
        self.get_logger().info('Motion recording started.')

    def stop_recording(self):
        """Stop recording and write frames as encoder counts to a timestamped .txt.

        Format: one frame per line, 20 space-separated values in URDF order
        (thumb → index → middle → ring → pinky), each value
        = rad → deg × (4096 / 360).
        Returns the saved Path on success, or None if nothing was recorded.
        """
        self.recording = False
        frames = self.recorded_frames
        self.recorded_frames = []
        n = len(frames)
        if n == 0:
            self.get_logger().warn('Recording stopped: no frames captured.')
            return None
        try:
            self.RECORDING_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            path = self.RECORDING_DIR / f'motion_{ts}.txt'
            with open(path, 'w') as f:
                for frame in frames:
                    counts = np.round(
                        np.rad2deg(frame) * self.DEG_TO_COUNTS
                    ).astype(int)
                    f.write(' '.join(str(c) for c in counts) + '\n')
            self.get_logger().info(f'Saved {n} frames → {path}')
            return path
        except OSError as e:
            self.get_logger().error(f'Recording save failed: {e}')
            return None

    def _try_load_tuning(self):
        """If self.TUNING_FILE exists, override known keys in scales/offsets."""
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
        """Persist current scales/offsets to self.TUNING_FILE (JSON)."""
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

    def transform_glove_to_v6(self, glove20):
        """
        Transform 20-D glove ergonomics to 20-D V6 joint angles.

        Pipeline:
          1. Extract per-finger glove values (4 each for thumb/index/middle/ring/pinky).
          2. Per-joint scale (multiplication, in degrees).
          3. Per-joint offset (addition, in degrees).
          4. Convert degrees → radians.
          5. Clip to V6 joint limits.
          6. EMA smoothing.

        Manus glove convention (per finger): val[0]=spread/CMC, val[1]=MCP
        flexion, val[2]=PIP flexion, val[3]=DIP flexion.

        V6 sign choices (joint_X0 has axis +Y in base_link; positive angle
        swings the finger toward +X / thumb side):
          - Index/Middle: positive sign for "spread out" (toward thumb).
          - Ring/Pinky: negative sign so anatomical "spread out" (toward -X,
            pinky side) maps to the joint's available negative range.
          - Pinky joint_41 (MCP) range [-90, +10] deg → flexion negated so
            positive glove flexion drives anatomical curl-into-palm.

        Thumb: V6 thumb mechanism differs from Allegro; a custom linear
        remap is used. All constants are starting points — calibrate.
        """
        # ====================================================================
        # Step 1: Extract finger values from glove data
        # ====================================================================
        thumb_vals  = np.array(glove20[0:4],   dtype=float)
        index_vals  = np.array(glove20[4:8],   dtype=float)
        middle_vals = np.array(glove20[8:12],  dtype=float)
        ring_vals   = np.array(glove20[12:16], dtype=float)
        pinky_vals  = np.array(glove20[16:20], dtype=float)

        # ====================================================================
        # Step 2: Per-joint scale (multiplication, in degrees), URDF order.
        #   Coefficients come from self.scales (mutable — sliders can override).
        #   joint_00 and joint_40 mix curl (val[1]) + spread (val[0]) since
        #   their v6 axes act as "CMC-like swings" rather than pure spreads.
        # ====================================================================
        s = self.scales
        angle_deg = np.array([
            # ----- Thumb (joint_00..03) -----
            s['t00_curl']   * thumb_vals[1] + s['t00_spread'] * thumb_vals[0],
            s['t01']        * thumb_vals[1],
            s['t02']        * thumb_vals[2],
            s['t03']        * thumb_vals[3],
            # ----- Index (joint_10..13) -----
            s['i10']        * index_vals[0],
            s['i11']        * index_vals[1],
            s['i12']        * index_vals[2],
            s['i13']        * index_vals[3],
            # ----- Middle (joint_20..23) -----
            s['m20']        * middle_vals[0],
            s['m21']        * middle_vals[1],
            s['m22']        * middle_vals[2],
            s['m23']        * middle_vals[3],
            # ----- Ring (joint_30..33) -----
            s['r30']        * ring_vals[0],
            s['r31']        * ring_vals[1],
            s['r32']        * ring_vals[2],
            s['r33']        * ring_vals[3],
            # ----- Pinky (joint_40..43) -----
            s['p40_curl']   * pinky_vals[1] + s['p40_spread'] * pinky_vals[0],
            s['p41']        * pinky_vals[1],
            s['p42']        * pinky_vals[2],
            s['p43']        * pinky_vals[3],
        ], dtype=float)

        # ====================================================================
        # Step 3: Per-joint offset (addition, in degrees), URDF order.
        #   Coefficients come from self.offsets (mutable — sliders can override).
        # ====================================================================
        o = self.offsets
        offset_deg = np.array([
            o['o_t00'], o['o_t01'], o['o_t02'], o['o_t03'],
            o['o_i10'], o['o_i11'], o['o_i12'], o['o_i13'],
            o['o_m20'], o['o_m21'], o['o_m22'], o['o_m23'],
            o['o_r30'], o['o_r31'], o['o_r32'], o['o_r33'],
            o['o_p40'], o['o_p41'], o['o_p42'], o['o_p43'],
        ], dtype=float)
        angle_deg = angle_deg + offset_deg

        # ====================================================================
        # Step 4: Convert degrees → radians
        # ====================================================================
        arr = np.deg2rad(angle_deg)

        # ====================================================================
        # Step 5: Clip to V6 joint limits
        # ====================================================================
        arr = np.clip(arr, V6_LOWER_LIMITS, V6_UPPER_LIMITS)

        # ====================================================================
        # Step 6: Apply EMA smoothing
        # ====================================================================
        if self.prev_arr is None:
            smoothed = arr.copy()
        else:
            smoothed = self.alpha * arr + (1.0 - self.alpha) * self.prev_arr

        # ====================================================================
        # Step 7: Anti-collision constraint between ring and pinky.
        #   Pinky's joint_41 (idx 17, MCP-like) must stay at least as negative
        #   as ring's joint_30 (idx 12, abduction) so the two fingers cannot
        #   overlap when the ring abducts toward the pinky side.
        #   Clamp after EMA so the smoothed (and stored) state is always
        #   constraint-compliant — next frame's EMA then starts from a
        #   compliant prev.
        # ====================================================================
        if smoothed[17] > smoothed[12]:
            smoothed[17] = smoothed[12]

        self.prev_arr = smoothed
        return smoothed

    # Type-string → (finger, motion) layout for the 20-D glove array.
    # Manus publisher emits 19 values (IndexMCPSpread is dropped by a bug in
    # ErgonomicsDataTypeToSide); we fill missing entries with 0.
    _FINGER_OFFSET = {'Thumb': 0, 'Index': 4, 'Middle': 8, 'Ring': 12, 'Pinky': 16}
    # Manus publisher inconsistently labels the spread channel: thumb uses
    # "ThumbMCPSpread" but middle/ring/pinky drop the MCP — "MiddleSpread",
    # "RingSpread", "PinkySpread" (see ErgonomicsDataTypeToString in
    # ManusDataPublisher.cpp). We accept both forms so all spread axes land
    # in slot 0. Index spread is never published (publisher bug).
    _MOTION_IDX = {
        'MCPSpread':  0,
        'Spread':     0,
        'MCPStretch': 1,
        'PIPStretch': 2,
        'DIPStretch': 3,
    }

    def _ergonomics_to_array(self, ergonomics_list):
        """Parse Manus ergonomics list by type-string into a 20-D array.
        Missing entries (e.g. IndexMCPSpread from the publisher bug) stay 0."""
        glove20 = np.zeros(20, dtype=float)
        for ergo in ergonomics_list:
            t = ergo.type  # e.g. "IndexDIPStretch", "ThumbMCPSpread"
            for finger, offset in self._FINGER_OFFSET.items():
                if t.startswith(finger):
                    motion = t[len(finger):]
                    if motion in self._MOTION_IDX:
                        glove20[offset + self._MOTION_IDX[motion]] = ergo.value
                    break
        return glove20

    def glove_callback(self, msg: ManusGlove):
        """Compute V6 qpos from incoming glove data and stash it."""
        if len(msg.ergonomics) == 0:
            return

        glove20 = self._ergonomics_to_array(msg.ergonomics)
        self.latest_glove20 = glove20
        self.latest_qpos = self.transform_glove_to_v6(glove20)

        if self.recording:
            self.recorded_frames.append(self.latest_qpos.copy())

        if not self._got_first:
            self._got_first = True
            self.get_logger().info(
                f'First glove command received ({len(msg.ergonomics)} ergonomics entries); driving sim.'
            )


def _setup_sapien_scene():
    """Create a fresh Sapien engine/renderer/scene matching hand_debug.py settings."""
    engine = sapien.Engine()
    renderer = sapien.VulkanRenderer()
    engine.set_renderer(renderer)

    scene_cfg = sapien.SceneConfig()
    scene_cfg.default_dynamic_friction = 1.0
    scene_cfg.default_static_friction = 1.0
    scene_cfg.default_restitution = 0.00
    scene_cfg.contact_offset = 0.02
    scene_cfg.enable_pcm = False
    # Bumped vs hand_debug.py defaults — v6's 20 DOFs / 5 fingers benefit
    # from a stiffer solver when self-collision is active.
    scene_cfg.solver_iterations = 50
    scene_cfg.solver_velocity_iterations = 2
    scene = engine.create_scene(scene_cfg)
    return engine, renderer, scene


def _tune_v6_drives(model: HandKinematicModel,
                    kp: float = 400.0, kd: float = 30.0,
                    force_limit: float = 10.0):
    """Override the PD gains HandKinematicModel set during construction.

    Defaults there are kp=400, kd=10. With self-collision enabled the v6
    hand gets sharp contact impulses; raising kd damps the resulting PD
    oscillation. Only used in physics mode (no effect under --kinematic).
    """
    for j in model.all_joints:
        j.set_drive_property(kp, kd, force_limit=force_limit)


# Distinct per-finger colors (RGBA) for visual debugging.
FINGER_COLORS = {
    'thumb':  [0.95, 0.30, 0.30, 1.0],   # red
    'index':  [0.30, 0.55, 0.95, 1.0],   # blue
    'middle': [0.30, 0.85, 0.40, 1.0],   # green
    'ring':   [0.95, 0.65, 0.20, 1.0],   # orange
    'pinky':  [0.75, 0.40, 0.85, 1.0],   # purple
}


def _set_link_color(link, rgba):
    """Recolor all visual shapes of a sapien link. Tries several API variants
    since sapien 2.x exposes different methods across builds."""
    try:
        bodies = link.get_visual_bodies()
    except AttributeError:
        return

    for body in bodies:
        # Variant 1: per-shape base color via material
        try:
            for shape in body.get_render_shapes():
                shape.material.set_base_color(rgba)
            continue
        except (AttributeError, TypeError):
            pass
        # Variant 2: body-level set_color
        try:
            body.set_color(rgba[:3])
        except (AttributeError, TypeError):
            pass


def _colorize_v6_fingers(articulation):
    """Apply FINGER_COLORS to each link whose name starts with a finger prefix."""
    for link in articulation.get_links():
        name = link.get_name()
        for finger, rgba in FINGER_COLORS.items():
            if name.startswith(finger + '_'):
                _set_link_color(link, rgba)
                break


class ErgonomicsBarViz:
    """Open3D dual-row bar chart for debugging the glove → V6 mapping.

    Two rows of 20 bars share the same X ordering (URDF order:
    thumb→index→middle→ring→pinky × [spread, MCP, PIP, DIP]):

      - Front row (z = -row_z_gap/2): raw Manus ergonomics in degrees.
        Bar height ∝ value * deg_scale (signed; bars can drop below 0).
      - Back row (z = +row_z_gap/2): V6 joint angle as fraction of its
        [lower, upper] range. Bar height ∝ fraction * frac_scale. A red
        horizontal line marks the fraction = 1.0 (upper limit) level.

    Reading the chart: bars at the same X give "raw sensor → joint range
    fraction". Saturated bottom bar (touching the red line or sitting at 0)
    means the joint hit a limit; flat top bar means the sensor isn't moving.
    """

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

        # Reference geometry: world frame + ground lines at y=0 for both rows,
        # plus a red line at y=frac_scale (= 100% of joint range) for the back row.
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(frame)

        x_min = -0.5
        x_max = 4 * finger_spacing + 3 * motion_spacing + 0.5
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

        # Initial: zero values
        self.update(np.zeros(20), np.zeros(20))

        self._first_render = True

    def _resize_bar(self, box, init_verts, signed_height):
        # Avoid degenerate box (causes NaN normals); keep a sliver.
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

    @staticmethod
    def print_legend():
        print("[ergo viz] Two rows along Z: front = ergonomics (deg, scaled), "
              "back = joint angle fraction (0..1 of [lower, upper]).")
        print("[ergo viz] Back-row red line = fraction 1.0 (joint upper limit).")
        print("[ergo viz] X order: thumb → index → middle → ring → pinky, "
              "each [spread, MCP, PIP, DIP].")
        print("[ergo viz] Color: thumb=red, index=blue, middle=green, "
              "ring=orange, pinky=purple.")


class TuningSliders:
    """Tkinter slider window for live editing the Step 2 scales and Step 3
    offsets used by ManusV6SimNode.transform_glove_to_v6.

    Each slider's `command` callback writes straight into the node's
    self.scales / self.offsets dicts (atomic dict-item set in CPython, so
    safe across the ROS callback thread). 5 columns laid out one per finger.
    Call `poll()` from the main loop to pump tkinter events.
    """

    # Per-finger groupings: (column title, list of (scale_key, label)).
    _SCALE_LAYOUT = [
        ('Thumb', [
            ('t00_curl',   '00 ← curl'),
            ('t00_spread', '00 ← spread'),
            ('t01',        '01 (curl)'),
            ('t02',        '02 (MCP)'),
            ('t03',        '03 (IP)'),
        ]),
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
        ('Pinky', [
            ('p40_curl',   '40 ← curl'),
            ('p40_spread', '40 ← spread'),
            ('p41',        '41 (MCP)'),
            ('p42',        '42 (PIP)'),
            ('p43',        '43 (DIP)'),
        ]),
    ]

    # Per-finger offset keys (URDF order within finger).
    _OFFSET_LAYOUT = [
        ('Thumb',  ['o_t00', 'o_t01', 'o_t02', 'o_t03']),
        ('Index',  ['o_i10', 'o_i11', 'o_i12', 'o_i13']),
        ('Middle', ['o_m20', 'o_m21', 'o_m22', 'o_m23']),
        ('Ring',   ['o_r30', 'o_r31', 'o_r32', 'o_r33']),
        ('Pinky',  ['o_p40', 'o_p41', 'o_p42', 'o_p43']),
    ]

    # Fonts (bumped up for readability)
    _FONT_HEADER  = ('TkDefaultFont', 13, 'bold')
    _FONT_SECTION = ('TkDefaultFont', 11, 'italic')
    _FONT_LABEL   = ('TkDefaultFont', 11)
    _FONT_SLIDER  = ('TkDefaultFont', 10)
    _FONT_BUTTON  = ('TkDefaultFont', 11, 'bold')

    def __init__(self, node,
                 scale_magnitude_max=3.5, scale_res=0.1,
                 offset_min=-15.0, offset_max=15.0, offset_res=1.0,
                 slider_length=200):
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
        self.root.title('V6 Retargeting Tuning')

        # (id(store), key) → tk.DoubleVar  — used by Reset to push values back into sliders
        self._slider_vars = {}

        outer = tk.Frame(self.root)
        outer.pack(padx=8, pady=8)

        for col_idx, (finger, scale_pairs) in enumerate(self._SCALE_LAYOUT):
            col = tk.LabelFrame(outer, text=finger,
                                font=self._FONT_HEADER,
                                padx=6, pady=6)
            col.grid(row=0, column=col_idx, padx=4, sticky='n')

            # --- Step 2 scales for this finger (sign locked to default) ---
            tk.Label(col, text='Step 2 scales  (sign locked)',
                     font=self._FONT_SECTION, fg='#555').pack(anchor='w')
            for key, label in scale_pairs:
                self._make_scale_slider(col, key, label)

            tk.Frame(col, height=2, bg='#bbb').pack(fill='x', pady=6)

            # --- Step 3 offsets for this finger (signed) ---
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

        # General status (Save / Reset feedback)
        self._status_var = tk.StringVar(value='')
        tk.Label(self.root, textvariable=self._status_var,
                 font=self._FONT_LABEL, fg='#0a0').pack(pady=(0, 4))

        # Start the periodic recording-status updater
        self._tick_rec_status()

    def _make_scale_slider(self, parent, key, label):
        """Step 2 scale slider — range chosen by the default's sign so the sign
        can never flip; only the magnitude moves."""
        default = self.node.scales[key]
        mag = self.scale_magnitude_max
        if default < 0:
            mn, mx = -mag, 0.0
        else:
            mn, mx = 0.0, mag
        self._make_slider(parent, self.node.scales, key, label,
                          mn, mx, self.scale_res)

    def _make_offset_slider(self, parent, key, label):
        """Step 3 offset slider — full bidirectional range."""
        self._make_slider(parent, self.node.offsets, key, label,
                          self.offset_min, self.offset_max, self.offset_res)

    def _make_slider(self, parent, store, key, label, mn, mx, res):
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
                self._flash_status(f'Recording → {path.name}')
            else:
                self._flash_status('Recording stopped (no frames).')
        else:
            self.node.start_recording()
            self._rec_button.config(text='■ Stop Record', fg='#080')

    def _tick_rec_status(self):
        """Periodically refresh the recording status label (frame count)."""
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
        """Pump tkinter events. Call from main loop alongside other viz."""
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            pass  # window closed

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


def _apply_qpos(model: HandKinematicModel, qpos_user, kinematic: bool):
    """Apply a user-order qpos to the sim model (kinematic snap or PD target)."""
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

  # Left glove instead (requires a v6_left config to exist)
  python v6_rule_based_retargeting.py --hand v6_left --side left
        """
    )
    parser.add_argument('--hand', type=str, default='v6_right',
                        help='Config name to load (default: v6_right)')
    parser.add_argument('--side', type=str, choices=['left', 'right'], default='right',
                        help='Glove side to subscribe to (default: right)')
    parser.add_argument('--no-kinematic', action='store_false', dest='kinematic',
                        help='Run physics-mode PD drive with self-collision response '
                             '(gains pre-tuned in _tune_v6_drives) instead of the '
                             'default kinematic snap. Physics mode may wobble on v6.')
    parser.add_argument('--no-ergo-viz', action='store_true',
                        help='Disable the Open3D ergonomics / joint-fraction debug window.')
    parser.add_argument('--no-tune', action='store_false', dest='tune',
                        help='Disable the tkinter live tuning slider window '
                             '(it is open by default).')
    args = parser.parse_args()

    # --------------------------------------------------------------------
    # Sapien scene + hand model + viewer
    # --------------------------------------------------------------------
    engine, renderer, scene = _setup_sapien_scene()

    config = get_config(args.hand)
    print(f"[v6_rule_based_retargeting] loading config: {config['name']}")

    model = HandKinematicModel.build_from_config(config, scene=scene, render=False)
    # Pose the hand so it's visible & oriented like in hand_debug.py
    model.hand.set_root_pose(sapien.Pose([0.0, 0.0, 0.35], [0.695, 0.0, -0.718, 0.0]))

    # Per-finger colors for visual clarity (URDF is uniform gray otherwise)
    _colorize_v6_fingers(model.hand)

    # Retune PD gains for v6 self-collision stability (only matters in physics mode)
    if not args.kinematic:
        _tune_v6_drives(model)

    # Fingertip markers (for visual sanity check)
    links = [info['link'] for info in config['fingertip_link']]
    offsets = [info['center_offset'] for info in config['fingertip_link']]
    model.initialize_keypoint(links, offsets)

    viewer_env = HandViewerEnv([model], scene=scene, renderer=renderer)

    # --------------------------------------------------------------------
    # Optional Open3D dual-row debug window
    #   front row: ergonomics (deg)
    #   back row : V6 joint angle as fraction of [lower, upper]
    # --------------------------------------------------------------------
    ergo_viz = None
    if not args.no_ergo_viz:
        ergo_viz = ErgonomicsBarViz(V6_LOWER_LIMITS, V6_UPPER_LIMITS)
        ErgonomicsBarViz.print_legend()

    # --------------------------------------------------------------------
    # ROS2 init + background spin
    # --------------------------------------------------------------------
    rclpy.init()
    node = ManusV6SimNode(side=args.side)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # --------------------------------------------------------------------
    # Optional tkinter slider window for live tuning
    # --------------------------------------------------------------------
    tuner = None
    if args.tune:
        try:
            tuner = TuningSliders(node)
            print("[v6_rule_based_retargeting] tuning slider window open.")
        except RuntimeError as e:
            print(f"[v6_rule_based_retargeting] failed to open tuner: {e}")

    mode_str = 'kinematic snap' if args.kinematic else 'physics / PD'
    print(f"[v6_rule_based_retargeting] running in {mode_str} mode. "
          f"Waiting for /manus_glove_{args.side} ...")

    # --------------------------------------------------------------------
    # Main viewer loop: read latest qpos and apply, then render one frame
    # --------------------------------------------------------------------
    try:
        while True:
            qpos = node.latest_qpos
            if qpos is not None:
                _apply_qpos(model, qpos, args.kinematic)

            if args.kinematic:
                # Skip physics: just update markers + render. Matches the
                # --kinematic branch in hand_debug.py.
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
