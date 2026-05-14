#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
Shared utilities for the rule-based retargeting scripts.

Pulled out of v6_rule_based_retargeting.py / allegro_rule_based_retargeting.py /
v6_rule_based_retargeting_realtime.py so the three platform-specific scripts
only carry their own joint limits, default tuning constants, and the
`transform_glove_to_<hand>` mapping. Everything generic — Sapien scene setup,
URDF-link colouring, Open3D ergonomics/joint-fraction debug viz, the tkinter
tuning slider window, and tuning-file JSON IO — lives here.

Provided:
  * setup_sapien_scene()
  * set_link_color(link, rgba)
  * colorize_articulation_by_finger(articulation, link_to_rgba)
  * tune_pd_drives(model, kp, kd, force_limit)
  * apply_qpos(model, qpos_user, kinematic)
  * load_tuning_file(path, scales, offsets)         — in-place dict update
  * save_tuning_file(path, scales, offsets)
  * ErgonomicsBarViz                                — N-channel dual-row bar
  * TuningSliders                                   — per-finger sliders +
                                                       Save/Load/Reset + optional
                                                       record + optional broadcast
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import sapien.core as sapien


# ----------------------------------------------------------------------
# Sapien helpers
# ----------------------------------------------------------------------
def setup_sapien_scene():
    """Create a fresh Sapien engine/renderer/scene matching hand_debug.py
    settings, with bumped solver iterations for many-DOF self-collision."""
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


def tune_pd_drives(model, kp: float = 400.0, kd: float = 30.0,
                   force_limit: float = 10.0):
    """Override the PD gains HandKinematicModel sets at construction
    (default kp=400, kd=10). Higher kd damps contact-impulse oscillation
    in many-DOF hands under self-collision."""
    for j in model.all_joints:
        j.set_drive_property(kp, kd, force_limit=force_limit)


def set_link_color(link, rgba):
    """Recolor every visual shape on a sapien link. Tries multiple API
    variants because sapien 2.x exposes different methods across builds."""
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


def colorize_articulation(articulation, link_classifier):
    """Apply per-link colors. `link_classifier` is a callable
    `link_name -> rgba | None`; None means "skip this link".
    """
    for link in articulation.get_links():
        rgba = link_classifier(link.get_name())
        if rgba is not None:
            set_link_color(link, rgba)


def apply_qpos(model, qpos_user, kinematic: bool):
    """Apply a user-order qpos to a HandKinematicModel — kinematic snap
    (set_qpos) or physics-mode PD target."""
    if kinematic:
        clipped = np.clip(qpos_user,
                          model.joint_lower_limit + 1e-3,
                          model.joint_upper_limit - 1e-3)
        qpos_sim = model.convert_user_order_to_sim_order(clipped)
        model.hand.set_qpos(qpos_sim)
        model.hand.set_qvel(np.zeros_like(qpos_sim))
    else:
        model.set_qpos_target(qpos_user)


# ----------------------------------------------------------------------
# Tuning-file JSON IO
# ----------------------------------------------------------------------
def load_tuning_file(path: Path, scales: dict, offsets: dict, logger=None):
    """Overlay disk values onto the given scales/offsets dicts IN PLACE.
    Unknown keys in the file are silently ignored. Returns True on success
    (file existed and parsed)."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        if logger is not None:
            logger.warn(f'Could not load {path}: {e}')
        return False

    for k, v in (data.get('scales') or {}).items():
        if k in scales:
            scales[k] = float(v)
    for k, v in (data.get('offsets') or {}).items():
        if k in offsets:
            offsets[k] = float(v)
    if logger is not None:
        logger.info(f'Loaded tuning from {path}')
    return True


def save_tuning_file(path: Path, scales: dict, offsets: dict, logger=None):
    """Write scales/offsets to a JSON file (alphabetically sorted keys)."""
    path = Path(path)
    try:
        with open(path, 'w') as f:
            json.dump({'scales': dict(scales), 'offsets': dict(offsets)},
                      f, indent=2, sort_keys=True)
        if logger is not None:
            logger.info(f'Saved tuning to {path}')
        return True
    except OSError as e:
        if logger is not None:
            logger.error(f'Save failed: {e}')
        return False


def reload_tuning_in_place(path: Path, scales: dict, offsets: dict,
                           default_scales: dict, default_offsets: dict,
                           logger=None):
    """Reset scales/offsets to factory defaults then overlay disk values.
    IMPORTANT: mutates `scales`/`offsets` in place — never reassign — so the
    slider closures that captured those dict references stay live."""
    scales.clear()
    scales.update(default_scales)
    offsets.clear()
    offsets.update(default_offsets)
    return load_tuning_file(path, scales, offsets, logger=logger)


# ----------------------------------------------------------------------
# Open3D ergonomics + joint-fraction debug window
# ----------------------------------------------------------------------
class ErgonomicsBarViz:
    """Open3D dual-row bar chart for debugging the glove → joint mapping.

    Two rows of N bars share the same X ordering:
      - Front row (z < 0): raw ergonomics values in degrees
        (bar height ∝ value * deg_scale, signed).
      - Back row (z > 0): joint angle as fraction of [lower, upper] range
        (bar height ∝ fraction * frac_scale). A red line at frac=1.0
        marks the joint upper limit.

    Reading: bars at the same X give "raw sensor → joint range fraction".
    Saturated back bar (touching the red line or sitting at 0) means the
    joint hit a limit; flat front bar means that sensor isn't moving.
    """

    def __init__(self, joint_lower, joint_upper, *,
                 fingers,             # list of finger names matching channel/4 layout
                 finger_rgb,          # dict[finger_name] -> [r, g, b]
                 motions=('spread', 'mcp', 'pip', 'dip'),
                 window_title='Ergonomics / Joint Fraction',
                 deg_scale: float = 0.01, frac_scale: float = 1.2,
                 bar_width: float = 0.35, bar_depth: float = 0.35,
                 init_height: float = 1.0,
                 finger_spacing: float = 2.6, motion_spacing: float = 0.5,
                 row_z_gap: float = 1.6,
                 window_size=(1200, 620)):
        self.joint_lower = np.asarray(joint_lower, dtype=float)
        self.joint_upper = np.asarray(joint_upper, dtype=float)
        self.joint_range = np.where(
            (self.joint_upper - self.joint_lower) > 1e-9,
            self.joint_upper - self.joint_lower, 1.0,
        )
        self.fingers = list(fingers)
        self.motions = list(motions)
        self.N = len(self.fingers) * len(self.motions)
        self.deg_scale = deg_scale
        self.frac_scale = frac_scale
        self.init_height = init_height

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_title,
                               width=window_size[0], height=window_size[1])

        self.boxes_top, self.init_verts_top = [], []
        self.boxes_bot, self.init_verts_bot = [], []
        self.bar_x = np.zeros(self.N)
        z_top = -row_z_gap / 2
        z_bot = +row_z_gap / 2

        for i in range(self.N):
            f_idx, m_idx = i // len(self.motions), i % len(self.motions)
            finger = self.fingers[f_idx]
            x = f_idx * finger_spacing + m_idx * motion_spacing
            self.bar_x[i] = x

            base = list(finger_rgb[finger][:3])
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

        # Reference geometry: world frame + two ground lines + frac=1 marker.
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(frame)

        x_min = -0.5
        x_max = ((len(self.fingers) - 1) * finger_spacing
                 + (len(self.motions) - 1) * motion_spacing + 0.5)
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

    def update(self, ergo_n, qpos_n):
        if ergo_n is not None and len(ergo_n) >= self.N:
            for i in range(self.N):
                h = float(ergo_n[i]) * self.deg_scale
                self._resize_bar(self.boxes_top[i], self.init_verts_top[i], h)
        if qpos_n is not None and len(qpos_n) >= self.N:
            q = np.asarray(qpos_n, dtype=float)
            normalized = (q - self.joint_lower) / self.joint_range
            for i in range(self.N):
                h = float(normalized[i]) * self.frac_scale
                self._resize_bar(self.boxes_bot[i], self.init_verts_bot[i], h)

    def render(self):
        if self._first_render:
            ctr = self.vis.get_view_control()
            ctr.set_lookat([self.bar_x.max() / 2 + 0.5, 0.8, 0.0])
            ctr.set_front([0.0, -0.3, 1.0])
            ctr.set_up([0.0, 1.0, 0.0])
            ctr.set_zoom(0.5 if self.N >= 20 else 0.55)
            self._first_render = False
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Tkinter slider window
# ----------------------------------------------------------------------
class TuningSliders:
    """Configurable slider window for Step 2 scales + Step 3 offsets.

    Wire it up with:
      * node                       — has .DEFAULT_SCALES, .DEFAULT_OFFSETS,
                                     .scales, .offsets, .save_tuning(),
                                     .reload_tuning(); optionally
                                     .start_recording(target_hz),
                                     .stop_recording(), .recording,
                                     .recorded_frames, ._rec_target_hz,
                                     .last_record_stats; and (for broadcast)
                                     .modbus with .start/.stop/.broadcasting.
      * scale_layout  : list of (finger_title, [(key, label), ...])
      * offset_layout : list of (finger_title, [offset_key, ...])
      * reversed_scale_keys : set of scale keys whose slider direction is
                              physically reversed (left=high, right=low)
      * with_load / with_record / with_broadcast : show those button rows
    """

    _FONT_HEADER  = ('TkDefaultFont', 13, 'bold')
    _FONT_SECTION = ('TkDefaultFont', 11, 'italic')
    _FONT_LABEL   = ('TkDefaultFont', 11)
    _FONT_SLIDER  = ('TkDefaultFont', 10)
    _FONT_BUTTON  = ('TkDefaultFont', 11, 'bold')

    def __init__(self, node, *,
                 scale_layout, offset_layout,
                 reversed_scale_keys=None,
                 scale_magnitude_max=3.5, scale_res=0.1,
                 offset_min=-90.0, offset_max=90.0, offset_res=5.0,
                 slider_length=200,
                 with_load=True,
                 with_record=True, default_record_hz=50,
                 with_broadcast=False, default_broadcast_hz=100,
                 title='Retargeting Tuning'):
        try:
            import tkinter as tk
        except ImportError as e:
            raise RuntimeError(
                "tkinter is required for the tuning window "
                "(try: sudo apt install python3-tk)"
            ) from e
        self.tk = tk
        self.node = node
        self.scale_layout = scale_layout
        self.offset_layout = offset_layout
        self.reversed_scale_keys = set(reversed_scale_keys or ())
        self.scale_magnitude_max = scale_magnitude_max
        self.scale_res = scale_res
        self.offset_min = offset_min
        self.offset_max = offset_max
        self.offset_res = offset_res
        self.slider_length = slider_length
        self.with_load = with_load
        self.with_record = with_record
        self.with_broadcast = with_broadcast

        self.root = tk.Tk()
        self.root.title(title)

        # (id(store), key) → (DoubleVar, store_dict)
        self._slider_vars = {}

        # ---- per-finger columns ------------------------------------
        outer = tk.Frame(self.root)
        outer.pack(padx=8, pady=8)
        for col_idx, (finger, scale_pairs) in enumerate(self.scale_layout):
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
            _, offset_keys = self.offset_layout[col_idx]
            for okey in offset_keys:
                self._make_offset_slider(col, okey, okey[2:])

        # ---- Row: Save / Load / Reset -------------------------------
        btn_row = tk.Frame(self.root)
        btn_row.pack(pady=(0, 4))
        tk.Button(btn_row, text='Save', font=self._FONT_BUTTON,
                  width=10, command=self._on_save).pack(side='left', padx=4)
        if self.with_load:
            tk.Button(btn_row, text='Load', font=self._FONT_BUTTON,
                      width=10, command=self._on_load).pack(side='left', padx=4)
        tk.Button(btn_row, text='Reset to defaults', font=self._FONT_BUTTON,
                  width=18,
                  command=self._reset_to_defaults).pack(side='left', padx=4)

        # ---- Optional row: Modbus broadcast -------------------------
        self._bc_button = None
        self._bc_hz_var = None
        self._bc_status_var = None
        if self.with_broadcast:
            bc_row = tk.Frame(self.root)
            bc_row.pack(pady=(0, 4))
            self._bc_button = tk.Button(
                bc_row, text='● Start Broadcast', font=self._FONT_BUTTON,
                width=20, fg='#a00',
                command=self._on_broadcast_toggle,
            )
            self._bc_button.pack(side='left', padx=4)
            tk.Label(bc_row, text='Hz:',
                     font=self._FONT_LABEL).pack(side='left', padx=(10, 2))
            self._bc_hz_var = tk.IntVar(value=int(default_broadcast_hz))
            tk.Spinbox(bc_row, from_=1, to=120, increment=1,
                       textvariable=self._bc_hz_var, width=5,
                       font=self._FONT_LABEL).pack(side='left', padx=2)
            self._bc_status_var = tk.StringVar(value='')
            tk.Label(bc_row, textvariable=self._bc_status_var,
                     font=self._FONT_LABEL, fg='#a00').pack(side='left', padx=4)

        # ---- Optional row: motion record ----------------------------
        self._rec_button = None
        self._rec_hz_var = None
        self._rec_status_var = None
        if self.with_record:
            rec_row = tk.Frame(self.root)
            rec_row.pack(pady=(0, 4))
            self._rec_button = tk.Button(
                rec_row, text='● Start Record', font=self._FONT_BUTTON,
                width=20, fg='#a00',
                command=self._on_record_toggle,
            )
            self._rec_button.pack(side='left', padx=4)
            tk.Label(rec_row, text='Hz:',
                     font=self._FONT_LABEL).pack(side='left', padx=(10, 2))
            self._rec_hz_var = tk.IntVar(value=int(default_record_hz))
            tk.Spinbox(rec_row, from_=1, to=120, increment=1,
                       textvariable=self._rec_hz_var, width=5,
                       font=self._FONT_LABEL).pack(side='left', padx=2)
            self._rec_status_var = tk.StringVar(value='')
            tk.Label(rec_row, textvariable=self._rec_status_var,
                     font=self._FONT_LABEL, fg='#a00').pack(side='left', padx=4)

        # ---- General flash status -----------------------------------
        self._status_var = tk.StringVar(value='')
        tk.Label(self.root, textvariable=self._status_var,
                 font=self._FONT_LABEL, fg='#0a0').pack(pady=(0, 4))

        # Start the periodic status updaters
        if self.with_broadcast:
            self._tick_bc_status()
        if self.with_record:
            self._tick_rec_status()

    # ----- slider builders -------------------------------------------
    def _make_scale_slider(self, parent, key, label):
        """Step 2 scale slider — sign range locked to the FACTORY default,
        with auto sign-correction on the loaded current value."""
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
        reverse = key in self.reversed_scale_keys
        self._make_slider(parent, self.node.scales, key, label,
                          mn, mx, self.scale_res, reverse=reverse)

    def _make_offset_slider(self, parent, key, label):
        self._make_slider(parent, self.node.offsets, key, label,
                          self.offset_min, self.offset_max, self.offset_res)

    def _make_slider(self, parent, store, key, label, mn, mx, res,
                     reverse=False):
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

        # Reversed slider: pass from_ > to to tk.Scale → drag direction flips.
        from_, to_ = (mx, mn) if reverse else (mn, mx)
        slider = tk.Scale(row, from_=from_, to=to_, resolution=res,
                          orient='horizontal', variable=var,
                          length=self.slider_length,
                          showvalue=True, command=on_change,
                          font=self._FONT_SLIDER)
        slider.pack(side='left')
        self._slider_vars[(id(store), key)] = (var, store)

    # ----- button handlers -------------------------------------------
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
            name = getattr(getattr(self.node, 'TUNING_FILE', None), 'name', '')
            self._flash_status(f'Saved → {name}' if name else 'Saved.')
        else:
            self._flash_status('Save failed (see console).')

    def _on_load(self):
        self.node.reload_tuning()
        for (store_id, key), (var, store) in self._slider_vars.items():
            if key in store:
                var.set(store[key])
        name = getattr(getattr(self.node, 'TUNING_FILE', None), 'name', '')
        self._flash_status(f'Loaded ← {name}' if name else 'Loaded.')

    def _on_record_toggle(self):
        if self.node.recording:
            path = self.node.stop_recording()
            self._rec_button.config(text='● Start Record', fg='#a00')
            if path is not None:
                stats = getattr(self.node, 'last_record_stats', None) or {}
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
            self._rec_status_var.set(
                f'REC {n} frames  (target {target:.0f} Hz)'
            )
        else:
            self._rec_status_var.set('')
        try:
            self.root.after(200, self._tick_rec_status)
        except self.tk.TclError:
            pass

    def _on_broadcast_toggle(self):
        mb = getattr(self.node, 'modbus', None)
        if mb is None:
            self._flash_status('Modbus not available (see startup logs).')
            return
        if mb.broadcasting:
            stats = mb.stop()
            self._bc_button.config(text='● Start Broadcast', fg='#a00')
            err = stats.get('errors', 0)
            err_str = f', {err} err' if err > 0 else ''
            self._flash_status(
                f"Stopped — sent {stats.get('n', 0)} @ "
                f"{stats.get('fps', 0.0):.1f} Hz{err_str}",
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
        mb = getattr(self.node, 'modbus', None)
        if mb is not None and mb.broadcasting:
            err = getattr(mb, 'send_errors', 0)
            err_str = f'  ({err} err)' if err > 0 else ''
            target = getattr(mb, '_bc_target_hz', 0.0)
            n = getattr(mb, 'n_sent', 0)
            self._bc_status_var.set(
                f'TX {n} frames  (target {target:.0f} Hz){err_str}'
            )
        else:
            self._bc_status_var.set('')
        try:
            self.root.after(200, self._tick_bc_status)
        except self.tk.TclError:
            pass

    # ----- utility ---------------------------------------------------
    def _flash_status(self, msg, ms=3000):
        self._status_var.set(msg)
        try:
            self.root.after(ms, lambda: self._status_var.set(''))
        except self.tk.TclError:
            pass

    def poll(self):
        """Pump tkinter events. Call from the main loop alongside other viz."""
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
