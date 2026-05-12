#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
Hand kinematic model and viewer environment for hand visualization & debug.

Provides:
- HandKinematicModel: kinematic model with FK + keypoint (fingertip) computation
- HandViewerEnv: multi-hand visualization with:
    * fingertip sphere markers (one per keypoint, at `link.world * center_offset`)
    * base-frame RGB axis markers (X=red, Y=green, Z=blue) anchored at the
      config's `base_link` entity — useful for verifying the GeoRT-template
      coordinate convention is matched across hands
- Collision filtering utilities for multi-hand scenes

CLI flags (entry point at bottom):
- `--hand <name> [<name> ...]` : load arbitrary config(s) (e.g., v6_right,
  allegro_right). Overrides legacy `--render-hand left|right|both`.
- `--kinematic` : snap qpos directly instead of PD-driving via physics. Use
  this for coordinate / offset verification — physics + self-collision can
  cause large hands (e.g., v6_right with 5 fingers) to flail wildly.
"""

import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer

from geort.utils.config_utils import get_config
from geort.utils.hand_utils import get_entity_by_name, get_active_joints, get_active_joint_indices


def set_articulation_collision_group(articulation, group_bit, ignore_bits):
    """
    Apply collision group/mask to ALL collision shapes of every link in the articulation.

    Compatible with SAPIEN builds where Link.set_collision_group(s) does not exist.
    """
    ALL = 0xFFFF
    collide_mask = ALL & (~ignore_bits)

    for link in articulation.get_links():
        # Some builds expose get_collision_shapes(); if not, skip safely
        try:
            shapes = link.get_collision_shapes()
        except AttributeError:
            shapes = []

        for shape in shapes:
            applied = False

            # Try common signatures on shape
            for args in ((group_bit, collide_mask, 0, 0),
                         (group_bit, collide_mask),
                         (group_bit,)):
                try:
                    shape.set_collision_group(*args)
                    applied = True
                    break
                except (AttributeError, TypeError):
                    try:
                        # Some versions use plural naming
                        shape.set_collision_groups(*args)
                        applied = True
                        break
                    except (AttributeError, TypeError):
                        pass

            # Optional: last-ditch attempt for filter-word style APIs (rare)
            if not applied:
                try:
                    # If these exist, they typically map to PhysX filter words
                    if hasattr(shape, "set_collision_filter_word0") and hasattr(shape, "set_collision_filter_word1"):
                        shape.set_collision_filter_word0(group_bit)
                        shape.set_collision_filter_word1(collide_mask)
                        applied = True
                except Exception:
                    pass

            # If still not applied, we just continue; not all shapes must have filters


class HandKinematicModel:
    def __init__(self,
                 scene=None,
                 render=False,
                 hand=None,
                 hand_urdf='',
                 n_hand_dof=16,
                 base_link='base_link',
                 joint_names=[],
                 kp=400.0,
                 kd=10):

        # If no shared scene passed in, create our own engine/renderer/scene
        self.engine = None
        if scene is None:
            engine = sapien.Engine()
            if render:
                renderer = sapien.VulkanRenderer()
                engine.set_renderer(renderer)
                print("Enable Render Mode.")
            else:
                renderer = None
            scene_config = sapien.SceneConfig()
            scene_config.default_dynamic_friction = 1.0
            scene_config.default_static_friction = 1.0
            scene_config.default_restitution = 0.00
            scene_config.contact_offset = 0.02
            scene_config.enable_pcm = False
            scene_config.solver_iterations = 25
            scene_config.solver_velocity_iterations = 1
            scene = engine.create_scene(scene_config)
            self.engine = engine
        # If a shared scene is provided, we assume its engine/renderer are already set up
        self.scene = scene
        self.renderer = self.scene.get_renderer() if hasattr(self.scene, "get_renderer") else None

        if hand is not None:
            self.hand = hand
        else:
            loader = scene.create_urdf_loader()
            self.hand = loader.load(hand_urdf)
            # Default pose (can be overridden in main)
            self.hand.set_root_pose(sapien.Pose([0, 0, 0.35], [0.695, 0, -0.718, 0]))

        self.pmodel = self.hand.create_pinocchio_model()

        # Base link
        self.base_link = get_entity_by_name(self.hand.get_links(), base_link)
        self.base_link_idx = self.hand.get_links().index(self.base_link)

        # DOF / joints
        self.all_joints = get_active_joints(self.hand, joint_names)
        all_limits = [joint.get_limits() for joint in self.all_joints]

        self.joint_names = joint_names
        self.user_idx_to_sim_idx = get_active_joint_indices(self.hand, joint_names)
        print("User-to-Sim Joint", self.user_idx_to_sim_idx)
        self.sim_idx_to_user_idx = [self.user_idx_to_sim_idx.index(i) for i in range(len(self.user_idx_to_sim_idx))]
        print("Sim-to-User Joint", self.sim_idx_to_user_idx)

        self.joint_lower_limit = np.array([l[0][0] for l in all_limits])
        self.joint_upper_limit = np.array([l[0][1] for l in all_limits])
        print(self.joint_lower_limit, self.joint_upper_limit)

        init_qpos = self.convert_user_order_to_sim_order((self.joint_lower_limit + self.joint_upper_limit) / 2)
        self.hand.set_qpos(init_qpos)
        self.hand.set_qvel(0.0 * init_qpos)
        self.qpos_target = init_qpos

        for i, joint in enumerate(self.all_joints):
            print(i, self.joint_names[i], joint, self.joint_lower_limit[i], self.joint_upper_limit[i])
            joint.set_drive_property(kp, kd, force_limit=10)

    def __del__(self):
        # If we created our own engine, allow cleanup; if shared, it's managed outside
        try:
            del self.engine
            del self.scene
        except Exception:
            pass

    def get_n_dof(self):
        return len(self.joint_lower_limit)

    def get_joint_limit(self):
        return self.joint_lower_limit, self.joint_upper_limit

    def initialize_keypoint(self, keypoint_link_names, keypoint_offsets):
        keypoint_links = [get_entity_by_name(self.hand.get_links(), link) for link in keypoint_link_names]
        print("Keypoint Links:", keypoint_links)
        print()

        keypoint_links_id_dict = {link_name: (self.hand.get_links().index(keypoint_links[i]), i)
                                  for i, link_name in enumerate(keypoint_link_names)}
        self.keypoint_links = keypoint_links
        self.keypoint_links_id_dict = keypoint_links_id_dict
        self.keypoint_offsets = np.array(keypoint_offsets)

    def convert_user_order_to_sim_order(self, qpos):
        return qpos[self.sim_idx_to_user_idx]

    def convert_sim_order_to_user_order(self, qpos_sim):
        """Convert simulator joint order -> user-specified joint order."""
        return qpos_sim[self.user_idx_to_sim_idx]

    def keypoint_from_qpos(self, qpos, ret_vec=False):
        """qpos in user order -> keypoints in base frame."""
        qpos = self.convert_user_order_to_sim_order(qpos)
        self.pmodel.compute_forward_kinematics(qpos)
        base_pose = self.pmodel.get_link_pose(self.base_link_idx)

        result = {}
        vec_result = []
        for m, (link_idx, i) in self.keypoint_links_id_dict.items():
            pose = self.pmodel.get_link_pose(link_idx)
            new_pose = sapien.Pose(
                p=pose.p + (pose.to_transformation_matrix()[:3, :3] @ self.keypoint_offsets[i].reshape(3, 1)).reshape(-1),
                q=pose.q,
            )
            x = (base_pose.inv() * new_pose).p  # convert to hand base frame
            vec_result.append(x)
            result[m] = x

        if ret_vec:
            return np.array(vec_result)
        return result

    @staticmethod
    def build_from_config(config, **kwargs):
        """Now accepts shared scene via kwargs['scene'] (optional)."""
        from pathlib import Path
        render = kwargs.get("render", False)
        urdf_path = config["urdf_path"]
        # Resolve "./xxx" against the glove_based/ directory (this file lives at
        # glove_based/geort/env/hand_debug.py), matching hand.py's behavior.
        if urdf_path.startswith("./"):
            glove_based_dir = Path(__file__).resolve().parent.parent.parent
            urdf_path = str(glove_based_dir / urdf_path[2:])
        n_hand_dof = len(config["joint_order"])
        base_link = config["base_link"]
        joint_order = config["joint_order"]
        scene = kwargs.get("scene", None)

        model = HandKinematicModel(hand_urdf=urdf_path,
                                   render=render,
                                   n_hand_dof=n_hand_dof,
                                   base_link=base_link,
                                   joint_names=joint_order,
                                   scene=scene)
        return model

    # keep for backward compatibility (unused below)
    def get_viewer_env(self):
        return HandViewerEnv([self], scene=self.scene, renderer=self.renderer)
    
    def set_qpos_target(self, qpos):
        """
        Set PD controller targets for this hand.
        qpos is expected in user order.
        """
        # joint limit clip
        qpos = np.clip(qpos, self.joint_lower_limit + 1e-3,
                       self.joint_upper_limit - 1e-3)
        # convert to sim order
        qpos = self.convert_user_order_to_sim_order(qpos)
        self.qpos_target = qpos
        # apply to each drive
        for i in range(len(qpos)):
            self.all_joints[i].set_drive_target(self.qpos_target[i])


class HandViewerEnv:
    """Supports multiple hand models in ONE scene + ONE viewer. Shows fingertip dots for each model."""

    def __init__(self, models, scene, renderer=None):
        self.models = models
        self.scene = scene

        # basic lights/ground/timestep (do once)
        self.scene.set_timestep(1 / 100.0)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_ground(altitude=0)

        # viewer
        if renderer is None:
            # Fallback if not passed (shouldn't happen when sharing)
            renderer = sapien.VulkanRenderer()
            self.scene.get_engine().set_renderer(renderer)
        self.viewer = Viewer(renderer)
        self.viewer.set_scene(self.scene)
        self.viewer.window.set_camera_position([0.1550926, -0.1623763, 0.7064089])
        self.viewer.window.set_camera_rotation([0.8716827, 0.3260138, 0.12817779, 0.3427167])
        self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        # per-model fingertip spheres
        self.tip_spheres = []  # list of list: tip_spheres[m_idx] -> [actors...]
        self._create_tip_spheres()

        # per-model base-frame axis markers (X=red, Y=green, Z=blue)
        self.axis_markers = []  # one actor per model
        self._create_axis_markers()

    def _create_tip_spheres(self):
        """Create visual-only spheres for each model's keypoints (one-time)."""
        if len(self.tip_spheres) > 0:
            return

        # Color palette per model (useful for left/right distinction)
        color_palette = [
            [1.0, 0.0, 0.0],  # red
            [0.0, 0.4, 1.0],  # blue-ish
            [0.0, 0.8, 0.2],  # green
            [1.0, 0.6, 0.0],  # orange
        ]
        radius = 0.005  # 5 mm

        for mi, model in enumerate(self.models):
            # If keypoints not initialized, create empty list and continue
            if not hasattr(model, "keypoint_links"):
                self.tip_spheres.append([])
                continue

            model_color = color_palette[mi % len(color_palette)]
            per_model_spheres = []
            for _ in model.keypoint_links:
                builder = self.scene.create_actor_builder()
                builder.add_sphere_visual(radius=radius, color=model_color)
                sphere = builder.build_static()  # purely visual; no collision
                per_model_spheres.append(sphere)
            self.tip_spheres.append(per_model_spheres)

    def _create_axis_markers(self, length=0.06, thickness=0.003):
        """Create one RGB axis-triad marker per model:
            X axis -> red box extending in +X
            Y axis -> green box extending in +Y
            Z axis -> blue box extending in +Z
        Each triad is anchored to the model's base_link entity (= the link
        named in the config's `base_link` field). This lets you visually
        confirm that two hands share the same coordinate convention (GeoRT
        template: X=palm normal, Y=palm->thumb, Z=palm->middle finger)."""
        if len(self.axis_markers) > 0:
            return
        for model in self.models:
            builder = self.scene.create_actor_builder()
            # X arrow (red): box extending in +X
            builder.add_box_visual(
                pose=sapien.Pose(p=[length / 2, 0, 0]),
                half_size=[length / 2, thickness, thickness],
                color=[1.0, 0.0, 0.0],
            )
            # Y arrow (green): box extending in +Y
            builder.add_box_visual(
                pose=sapien.Pose(p=[0, length / 2, 0]),
                half_size=[thickness, length / 2, thickness],
                color=[0.0, 1.0, 0.0],
            )
            # Z arrow (blue): box extending in +Z
            builder.add_box_visual(
                pose=sapien.Pose(p=[0, 0, length / 2]),
                half_size=[thickness, thickness, length / 2],
                color=[0.0, 0.0, 1.0],
            )
            self.axis_markers.append(builder.build_static(name="axis_marker"))

    def _update_axis_markers(self):
        """Each frame: snap each marker to its model's base_link world pose."""
        for mi, model in enumerate(self.models):
            if mi >= len(self.axis_markers):
                continue
            # model.base_link is the link entity matching config["base_link"]
            # (resolved in HandKinematicModel.__init__).
            self.axis_markers[mi].set_pose(model.base_link.get_pose())

    def _update_tip_positions(self):
        """Each frame: place each model's fingertip spheres at the WORLD-frame
        position `link.world_pose * center_offset` — the exact point GeoRT
        treats as the keypoint target during training.

        NOTE: we do NOT use `keypoint_from_qpos`, which returns coordinates in
        the base_link LOCAL frame; feeding those into `set_pose` (world frame)
        would put markers offset from the hand by the root_pose translation.
        Reading `link.get_pose()` directly gives the world pose that already
        accounts for the articulation root_pose + chain FK.
        """
        for mi, model in enumerate(self.models):
            if not hasattr(model, "keypoint_links"):
                continue
            # lazily create if missing (e.g., keypoints inited after viewer)
            if mi >= len(self.tip_spheres) or len(self.tip_spheres[mi]) == 0:
                self._create_tip_spheres()
                if mi >= len(self.tip_spheres) or len(self.tip_spheres[mi]) == 0:
                    continue

            for i, (link, offset) in enumerate(zip(model.keypoint_links,
                                                   model.keypoint_offsets)):
                lp = link.get_pose()
                R = lp.to_transformation_matrix()[:3, :3]
                tip_world = lp.p + R @ np.asarray(offset)
                self.tip_spheres[mi][i].set_pose(sapien.Pose(p=tip_world))

    def update(self):
        self.scene.step()
        self._update_tip_positions()
        self._update_axis_markers()
        self.scene.update_render()
        self.viewer.render()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Visualize hand(s) with fingertip markers")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--render-hand",
        choices=["left", "right", "both"],
        default="both",
        help="Allegro hand(s) to visualize (legacy default). Ignored if --hand is given.",
    )
    group.add_argument(
        "--hand", nargs="+",
        help="Config name(s) to load (e.g., 'v6_right', 'allegro_right v6_right'). "
             "Overrides --render-hand.",
    )
    parser.add_argument(
        "--kinematic", action="store_true",
        help="Snap qpos directly without physics — avoids PD/self-collision wobble. "
             "Recommended for coordinate-frame and fingertip-offset verification: "
             "hands with many DOFs and tight finger packing (e.g., v6_right's 5 "
             "fingers / 20 DOFs) thrash under PD + self-collision otherwise.",
    )
    parser.add_argument(
        "--pose", choices=["random", "zero", "mid"], default="random",
        help="Static pose mode (default: random cycles every 30 steps). "
             "'zero' = all joint angles = 0 (fully extended fingers; best for "
             "checking fingertip offsets). 'mid' = midpoint of each joint's "
             "limits.",
    )
    args = parser.parse_args()

    # Resolve which configs to load.
    if args.hand:
        config_names = list(args.hand)
    else:
        config_names = []
        if args.render_hand in ("left", "both"):
            config_names.append("allegro_left")
        if args.render_hand in ("right", "both"):
            config_names.append("allegro_right")

    # 1) Shared engine/renderer/scene
    engine = sapien.Engine()
    renderer = sapien.VulkanRenderer()
    engine.set_renderer(renderer)

    scene_cfg = sapien.SceneConfig()
    scene_cfg.default_dynamic_friction = 1.0
    scene_cfg.default_static_friction = 1.0
    scene_cfg.default_restitution = 0.00
    scene_cfg.contact_offset = 0.02
    scene_cfg.enable_pcm = False
    scene_cfg.solver_iterations = 25
    scene_cfg.solver_velocity_iterations = 1
    shared_scene = engine.create_scene(scene_cfg)

    # 2) Load configs, build models, initialize keypoints — all generic in a single loop
    configs = [get_config(name) for name in config_names]
    print(f"[hand_debug] loading {len(configs)} hand(s): {[c['name'] for c in configs]}")

    models = []
    for cfg in configs:
        m = HandKinematicModel.build_from_config(cfg, scene=shared_scene, render=False)
        m.hand.set_root_pose(sapien.Pose([0.0, 0.0, 0.35], [0.695, 0.0, -0.718, 0.0]))
        models.append(m)

    # If exactly two hands, filter collisions between them (keeps the original
    # left/right behavior; harmless for other 2-config combinations).
    if len(models) == 2:
        GROUP_A = 1 << 0
        GROUP_B = 1 << 1
        set_articulation_collision_group(models[0].hand, GROUP_A, ignore_bits=GROUP_B)
        set_articulation_collision_group(models[1].hand, GROUP_B, ignore_bits=GROUP_A)

    for m, cfg in zip(models, configs):
        links = [info["link"] for info in cfg["fingertip_link"]]
        offsets = [info["center_offset"] for info in cfg["fingertip_link"]]
        m.initialize_keypoint(links, offsets)

    # 3) Viewer env (creates fingertip spheres for every model)
    viewer_env = HandViewerEnv(models, scene=shared_scene, renderer=renderer)

    # 4) Control loop — re-randomize qpos targets every 30 sim steps
    model_infos = []
    for m in models:
        lo, hi = m.get_joint_limit()
        model_infos.append({"model": m, "lower": lo, "upper": hi})

    # Helper: compute initial qpos per model based on --pose choice.
    def _qpos_for_pose(pose_choice, lower, upper, dof):
        if pose_choice == "zero":
            return np.zeros(dof)
        elif pose_choice == "mid":
            return (np.asarray(lower) + np.asarray(upper)) / 2
        else:  # "random" — initial random sample (the loop below will keep cycling)
            return np.random.uniform(0, 1, dof) * (upper - lower - 1e-7) + lower + 1e-7

    # Apply initial pose to every model before entering the loop.
    for info in model_infos:
        mdl = info["model"]
        lower, upper = info["lower"], info["upper"]
        qpos_user = _qpos_for_pose(args.pose, lower, upper, mdl.get_n_dof())
        qpos_user = np.clip(qpos_user,
                            np.asarray(lower) + 1e-3,
                            np.asarray(upper) - 1e-3)
        if args.kinematic:
            qpos_sim = mdl.convert_user_order_to_sim_order(qpos_user)
            mdl.hand.set_qpos(qpos_sim)
            mdl.hand.set_qvel(np.zeros_like(qpos_sim))
        else:
            mdl.set_qpos_target(qpos_user)

    # Main loop: render and (if pose == "random") re-sample targets periodically.
    #   physics mode   : viewer_env.update() steps physics; PD drives joints
    #                    toward target. Fluid for allegro, wobbles for 5-finger v6.
    #   kinematic mode : skip scene.step(); set_qpos snaps the articulation
    #                    instantly. Use this for verification (coord frame,
    #                    fingertip offsets) — the rendered pose is exactly
    #                    what GeoRT's FK would compute.
    #   pose != random : static — skip the resampling block entirely so the
    #                    hand holds one pose indefinitely. Ideal for offset
    #                    checks (`--pose zero` gives fully-extended fingers).
    steps = 0
    while True:
        if args.kinematic:
            viewer_env._update_tip_positions()
            viewer_env._update_axis_markers()
            viewer_env.scene.update_render()
            viewer_env.viewer.render()
        else:
            viewer_env.update()
        steps += 1
        if steps % 30 == 0 and args.pose == "random":
            for info in model_infos:
                mdl, lower, upper = info["model"], info["lower"], info["upper"]
                dof = mdl.get_n_dof()
                targets = np.random.uniform(0, 1, dof) * (upper - lower - 1e-7) + lower + 1e-7
                if args.kinematic:
                    # Clip slightly inside limits (matches set_qpos_target's
                    # epsilon margin) then snap via sapien articulation API.
                    targets_clipped = np.clip(targets,
                                              np.asarray(lower) + 1e-3,
                                              np.asarray(upper) - 1e-3)
                    qpos_sim = mdl.convert_user_order_to_sim_order(targets_clipped)
                    mdl.hand.set_qpos(qpos_sim)
                    mdl.hand.set_qvel(np.zeros_like(qpos_sim))
                else:
                    mdl.set_qpos_target(targets)
