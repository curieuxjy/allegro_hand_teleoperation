#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
Static URDF viewer for quick visual inspection.

Features:
- Colors each link differently for visual distinction
- Prints joint/link tree with limits
- No motion, no controllers - just a static render loop
- Tiny RGB axes drawn at the config's base_link (for v6 this is `virtual_base`,
  so the axes appear at the GeoRT-aligned palm/middle-finger reference frame)

CLI:
  --hand <name>   config name (e.g., 'allegro_right', 'v6_right'). The URDF
                  path is read from the config, so any hand defined under
                  glove_based/geort/config/ is supported automatically.
  --handness {left,right}   legacy shortcut for allegro_{left,right}.
                  Ignored if --hand is given.
"""

import argparse
import colorsys
import random
from pathlib import Path

import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer

from geort.utils.config_utils import get_config
from geort.utils.hand_utils import get_entity_by_name


def make_scene(enable_render=True):
    """Create engine, scene, and (optionally) renderer."""
    engine = sapien.Engine()
    renderer = None
    if enable_render:
        renderer = sapien.VulkanRenderer()
        engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_ambient_light([0.6, 0.6, 0.6])
    scene.add_directional_light([0.5, 1.0, -1.0], [0.6, 0.6, 0.6], shadow=False)
    scene.add_ground(altitude=0.0)
    return engine, scene, renderer


def load_articulation(scene, urdf_path, fix_root=True):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root
    art = loader.load(urdf_path)
    if art is None:
        raise FileNotFoundError(f"Failed to load URDF: {urdf_path}")
    art.set_root_pose(sapien.Pose([0, 0, 0.3]))
    return art


def random_color(alpha=1.0):
    """Generate a random color with good saturation and value."""
    h = random.random()
    s = 0.6
    v = 0.9
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return [r, g, b, alpha]


def color_links(renderer, articulation):
    """
    Color each link with a random color.

    Tries multiple APIs depending on SAPIEN version:
    - link.set_visual_material(material) (newer)
    - Fallback: iterate visual bodies > render shapes
    """
    if renderer is None:
        return
    for link in articulation.get_links():
        mat = renderer.create_material()
        mat.set_base_color(random_color(1.0))
        # Try direct API
        try:
            link.set_visual_material(mat)
            continue
        except Exception:
            pass
        # Fallback path
        try:
            vbs = link.get_visual_bodies()
        except Exception:
            vbs = []
        for vb in vbs:
            try:
                for rs in vb.get_render_shapes():
                    rs.set_material(mat)
            except Exception:
                pass


def print_tree(articulation):
    """Print articulation summary including links and joints with their limits."""
    print("\n=== Articulation Summary ===")
    print(f"Name: {articulation.get_name()}")
    links = articulation.get_links()

    # get_joints() may include fixed; get_active_joints() excludes fixed
    try:
        joints = articulation.get_joints()
        if not joints:
            joints = articulation.get_active_joints()
    except Exception:
        joints = articulation.get_active_joints()

    print(f"Links ({len(links)}):")
    for i, l in enumerate(links):
        print(f"  [{i:02d}] {l.get_name()}")

    print(f"\nJoints ({len(joints)}):")
    for j in joints:
        try:
            name = j.get_name()
        except Exception:
            name = "<unnamed>"
        try:
            parent = j.get_parent_link().get_name()
        except Exception:
            parent = "<root or N/A>"
        try:
            child = j.get_child_link().get_name()
        except Exception:
            child = "<N/A>"

        # Limits (if present)
        lim_txt = ""
        try:
            lims = j.get_limits()
            flat = []
            for pair in lims:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    flat.append(f"{pair[0]:.3f}..{pair[1]:.3f}")
            if flat:
                lim_txt = f"  limits: {', '.join(flat)}"
        except Exception:
            pass

        # Type (if available)
        typ_txt = ""
        try:
            typ_txt = f"  type: {j.get_joint_type()}"
        except Exception:
            pass

        print(f"  - {name}: {parent} -> {child}{typ_txt}{lim_txt}")


def place_tiny_axes(scene, pose, length=0.03, radius=0.0015):
    """
    Draw tiny RGB coordinate axes at the given pose using capsules.

    Args:
        scene: SAPIEN scene
        pose: Pose to place the axes at
        length: Length of each axis
        radius: Radius of each axis capsule
    """
    # X axis (red)
    scene.add_capsule_visual(
        pose=pose * sapien.Pose([length / 2, 0, 0], [0.7071, 0, 0.7071, 0]),
        radius=radius, half_length=length / 2, color=[1, 0, 0]
    )
    # Y axis (green)
    scene.add_capsule_visual(
        pose=pose * sapien.Pose([0, length / 2, 0], [0, 0.7071, 0.7071, 0]),
        radius=radius, half_length=length / 2, color=[0, 1, 0]
    )
    # Z axis (blue)
    scene.add_capsule_visual(
        pose=pose * sapien.Pose([0, 0, length / 2], [0, 0, 0, 1]),
        radius=radius, half_length=length / 2, color=[0, 0, 1]
    )


def main():
    """Main entry point for the static URDF viewer."""
    parser = argparse.ArgumentParser(description="Static URDF link/joint inspector with coloring")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hand", type=str, default=None,
                       help="Config name (e.g., 'allegro_right', 'v6_right'). "
                            "Loads URDF + base_link from glove_based/geort/config/<name>.json.")
    group.add_argument("--handness", type=str, choices=["left", "right"], default=None,
                       help="Legacy shortcut: maps to allegro_left / allegro_right.")
    parser.add_argument("--no-fix-root", action="store_true", help="Do not fix root link")
    parser.add_argument("--seed", type=int, default=0, help="Color randomness seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Resolve which config to load.
    if args.hand is not None:
        config_name = args.hand
    elif args.handness is not None:
        config_name = f"allegro_{args.handness}"
    else:
        parser.error("Pass either --hand <name> or --handness {left,right}.")

    config = get_config(config_name)

    # URDF path: configs use "./xxx" relative to glove_based/. Resolve to absolute
    # path the same way env/hand.py does so the loader can find the file from any CWD.
    urdf_path = config["urdf_path"]
    if urdf_path.startswith("./"):
        glove_based_dir = Path(__file__).resolve().parent.parent.parent
        urdf_path = str(glove_based_dir / urdf_path[2:])

    engine, scene, renderer = make_scene(enable_render=True)
    art = load_articulation(scene, urdf_path, fix_root=not args.no_fix_root)

    # Color each link
    color_links(renderer, art)

    # Console summary
    print_tree(art)

    # Draw tiny axes at the config's base_link (so v6_right's virtual_base shows
    # the GeoRT-aligned frame, not the wrist-mounted base_link). Fall back to the
    # first link if the named base_link can't be resolved.
    try:
        base_link_name = config.get("base_link", None)
        base_link = (get_entity_by_name(art.get_links(), base_link_name)
                     if base_link_name else None) or art.get_links()[0]
        base_pose = base_link.get_pose()
        place_tiny_axes(scene, base_pose)
        print(f"\nAxes drawn at link: {base_link.get_name()}")
    except Exception as e:
        print(f"\n[warn] could not draw axes at base_link: {e}")

    # Viewer
    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.window.set_camera_position([0.4, -0.6, 0.5])
    viewer.window.set_camera_rotation([0.87, 0.32, 0.12, 0.34])
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1.0)

    print("\nViewer running. Close the window to exit.")
    while not viewer.closed:
        scene.update_render()
        viewer.render()


if __name__ == "__main__":
    main()
