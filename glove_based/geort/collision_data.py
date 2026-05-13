"""
Generate self-collision dataset for training the collision classifier.

For each random qpos sample drawn within the joint limits:
  1. Set the articulation's qpos in the sapien scene.
  2. Step the scene once (tiny dt, no gravity) so PhysX computes contacts.
  3. Inspect contacts; flag the sample as a collision if any pair of
     non-adjacent links of the articulation has at least one contact point
     with negative separation (true penetration).
  4. Record (qpos_user_order, 0_or_1_label).

This is the training data for the collision classifier C in the GeoRT paper:
the classifier learns C(qpos) ≈ P(self-collision | qpos), and trainer.py uses
it as a frozen differentiable proxy for the sapien-based check while training
the retargeting (IK) network.

Output: data/{hand_name}_collision.npz
  qpos:  (N, dof) float32 — joint configurations in USER order
  label: (N,)     int32   — 1 if self-collision detected, else 0

Usage:
  python glove_based/geort/collision_data.py --hand v6_right.json
"""
import argparse
import os
from pathlib import Path

import numpy as np
import sapien.core as sapien
from tqdm import tqdm

from geort.utils.config_utils import get_config
from geort.env.hand import HandKinematicModel
from geort.utils.path import get_data_root


def enable_self_collision_groups(articulation):
    """Force every link's collision shapes into the same collision group so
    PhysX permits intra-articulation contacts (only adjacent parent-child
    pairs remain filtered out automatically). Tries several API signatures
    because sapien revisions vary.
    """
    GROUP = 0x0001
    MASK = 0xFFFF  # collide with everything (including own group)
    for link in articulation.get_links():
        try:
            shapes = link.get_collision_shapes()
        except AttributeError:
            shapes = []
        for shape in shapes:
            applied = False
            for args in ((GROUP, MASK, 0, 0), (GROUP, MASK), (GROUP,)):
                try:
                    shape.set_collision_group(*args)
                    applied = True
                    break
                except (AttributeError, TypeError):
                    try:
                        shape.set_collision_groups(*args)
                        applied = True
                        break
                    except (AttributeError, TypeError):
                        pass
            # If the API path isn't found, fall through silently; the default
            # URDF-loader groups may already permit self-collision in this
            # sapien build.


def check_self_collision(scene, link_set, min_penetration: float = 0.002):
    """Return True if any contact between two articulation links has a contact
    point penetrating deeper than `min_penetration` (meters).

    Why a threshold? URDF meshes typically have sub-mm overlaps at joint
    boundaries (e.g., a proximal phalanx mesh designed to nest into the base
    mesh for visual continuity). PhysX flags these as penetrations of ~0.1
    to ~1.5 mm even at the zero/rest pose, which would falsely label most
    samples as colliding. Real self-collisions (a fingertip pressing into
    another finger or palm) are typically multi-mm to multi-cm penetrations,
    so a 2 mm threshold cleanly separates the two regimes.
    """
    contacts = scene.get_contacts()
    for c in contacts:
        if c.actor0 not in link_set or c.actor1 not in link_set:
            continue
        for p in c.points:
            if p.separation < -min_penetration:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate (qpos, self-collision-label) dataset for the collision classifier."
    )
    parser.add_argument("--hand", type=str, required=True,
                        help="Hand config name (e.g., v6_right.json)")
    parser.add_argument("--n_samples", type=int, default=1000000,
                        help="Number of random qpos samples to label (default: 1000000)")
    parser.add_argument("--min_penetration", type=float, default=0.002,
                        help="Penetration depth (m) above which a contact counts as a "
                             "real self-collision; sub-threshold contacts are treated "
                             "as URDF mesh-design artifacts (default: 0.002 = 2mm)")
    parser.add_argument("--strategy", type=str, default="balanced",
                        choices=["random", "single_joint", "few_joints", "balanced"],
                        help="Sampling strategy. 'random' = each joint uniformly within its "
                             "limits (high collision rate, current default). 'single_joint' "
                             "= zero pose with ONE joint perturbed (mostly collision-free). "
                             "'few_joints' = zero pose with K joints perturbed where K is "
                             "uniformly sampled in [1, dof]. 'balanced' (default) = mix of "
                             "the three to give a roughly 30-50%% collision rate.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    config = get_config(args.hand)
    hand_name = config["name"]

    # Build hand in a fresh scene. HandKinematicModel creates its own
    # engine/scene under the hood.
    hand_model = HandKinematicModel.build_from_config(config, render=False)
    art = hand_model.hand
    scene = hand_model.scene

    # Make the scene as static-friendly as possible for kinematic collision queries:
    # tiny timestep so dynamics can't drift, no gravity so the hand doesn't fall.
    scene.set_timestep(1e-4)
    try:
        scene.set_gravity([0.0, 0.0, 0.0])
    except Exception:
        pass

    # Ensure self-collision is detectable (default URDF loader sometimes
    # filters intra-articulation contacts).
    enable_self_collision_groups(art)

    # Pre-fetch link entity set for fast membership test in the loop.
    link_set = set(art.get_links())

    # Joint limits (USER order; matches saved qpos convention)
    lo, hi = hand_model.get_joint_limit()
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    dof = hand_model.get_n_dof()

    def sample_qpos(strategy):
        """Build one qpos sample according to the chosen sampling strategy.
        All strategies stay within joint limits."""
        if strategy == "random":
            # Each joint independently uniform within limits — most uniform
            # coverage of joint space but heavily biased toward collisions.
            return np.random.uniform(0.0, 1.0, dof) * (hi - lo - 1e-6) + lo + 1e-6
        if strategy == "single_joint":
            # Zero pose with exactly ONE joint moved within its full range —
            # mostly collision-free since other links stay at rest.
            qpos = np.clip(np.zeros(dof), lo + 1e-6, hi - 1e-6)
            j = np.random.randint(dof)
            qpos[j] = np.random.uniform(lo[j] + 1e-6, hi[j] - 1e-6)
            return qpos
        if strategy == "few_joints":
            # Zero pose with K joints perturbed, K uniform in [1, dof]. Smooth
            # interpolation between single_joint (K=1) and random (K=dof).
            K = np.random.randint(1, dof + 1)
            qpos = np.clip(np.zeros(dof), lo + 1e-6, hi - 1e-6)
            active = np.random.choice(dof, K, replace=False)
            for j in active:
                qpos[j] = np.random.uniform(lo[j] + 1e-6, hi[j] - 1e-6)
            return qpos
        # "balanced": evenly mix the three above
        r = np.random.random()
        if r < 1 / 3:
            return sample_qpos("single_joint")
        if r < 2 / 3:
            return sample_qpos("few_joints")
        return sample_qpos("random")

    qpos_arr = np.zeros((args.n_samples, dof), dtype=np.float32)
    label_arr = np.zeros(args.n_samples, dtype=np.int32)

    n_collisions = 0
    for i in tqdm(range(args.n_samples), desc="Sampling collision labels"):
        qpos_user = sample_qpos(args.strategy)
        qpos_sim = hand_model.convert_user_order_to_sim_order(qpos_user)

        art.set_qpos(qpos_sim)
        art.set_qvel(np.zeros_like(qpos_sim))
        # Keep PD drive targets in sync so the joints don't immediately try to
        # move to a stale target during the step.
        for j_idx, joint in enumerate(hand_model.all_joints):
            joint.set_drive_target(qpos_sim[j_idx])

        scene.step()
        in_collision = check_self_collision(scene, link_set,
                                            min_penetration=args.min_penetration)

        qpos_arr[i] = qpos_user.astype(np.float32)
        label_arr[i] = int(in_collision)
        if in_collision:
            n_collisions += 1

    pct = 100.0 * n_collisions / max(args.n_samples, 1)
    print(f"\nGenerated {args.n_samples} samples")
    print(f"  in collision:    {n_collisions} ({pct:.2f}%)")
    print(f"  collision-free:  {args.n_samples - n_collisions} ({100.0 - pct:.2f}%)")
    if n_collisions == 0:
        print("\n[warning] No self-collisions detected. This likely means your sapien")
        print("          build filters out intra-articulation contacts even after the")
        print("          collision-group remap. Inspect a few obviously-bad qpos manually,")
        print("          or fall back to a geometric proxy (pairwise link distance).")

    out_dir = Path(get_data_root())
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f"{hand_name}_collision.npz"
    np.savez(out_path, qpos=qpos_arr, label=label_arr)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
