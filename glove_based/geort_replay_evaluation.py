import argparse
import numpy as np

from geort.replay_mocap import ReplayMocap
from geort.env.hand import HandKinematicModel
from geort.env.hand_static import color_links
from geort import load_model, get_config


def disable_self_collision(articulation):
    """Filter out all intra-articulation collisions by putting every link in
    the same group with a mask that EXCLUDES that group. Hand-vs-other-objects
    collisions still work; only hand-vs-itself is suppressed."""
    GROUP = 0x0001
    MASK = 0xFFFE  # all bits except GROUP
    for link in articulation.get_links():
        try:
            shapes = link.get_collision_shapes()
        except AttributeError:
            shapes = []
        for shape in shapes:
            for sig in ((GROUP, MASK, 0, 0), (GROUP, MASK), (GROUP,)):
                try:
                    shape.set_collision_group(*sig)
                    break
                except (AttributeError, TypeError):
                    try:
                        shape.set_collision_groups(*sig)
                        break
                    except (AttributeError, TypeError):
                        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand', type=str, default='allegro')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--data', type=str, default='human')
    parser.add_argument('--use_last', action='store_true',
                        help='Load last checkpoint instead of best checkpoint (default: best)')
    parser.add_argument('--kinematic', action='store_true',
                        help="Snap qpos directly each frame instead of PD-driving through "
                             "physics. Disables self-collision wobble — useful for inspecting "
                             "what the IK actually predicts. Default off (uses physics).")
    # Physics-mode stability knobs (ignored under --kinematic).
    parser.add_argument('--kp', type=float, default=400.0,
                        help='PD position gain (default 400). Lower = softer spring.')
    parser.add_argument('--kd', type=float, default=40.0,
                        help='PD velocity gain (default 40 — bumped from in-repo default 10 '
                             'for dense 5-finger hands; gives damping ratio ~1.0 with kp=400).')
    parser.add_argument('--force_limit', type=float, default=10.0,
                        help='Per-joint force limit Nm (default 10). Caps how hard PD can push.')
    parser.add_argument('--no_self_collision', action='store_true',
                        help='Disable intra-articulation collision detection. The hand still '
                             'collides with external objects (e.g. ground), but PhysX no longer '
                             'fights self-overlap — eliminates the most common cause of '
                             'replay-mode divergence.')

    args = parser.parse_args()

    # GeoRT Model.
    epoch_to_load = 0 if args.use_last else 'best'
    model = load_model(args.ckpt, epoch=epoch_to_load)


    # Motion Capture.
    mocap = ReplayMocap(args.data)

    # Robot Simulation.
    config = get_config(args.hand)
    hand = HandKinematicModel.build_from_config(config, render=True)
    # Tint each link a distinct random color so finger boundaries / collisions
    # are easy to read visually (uses the same helper as hand_static.py).
    color_links(hand.get_renderer(), hand.hand)

    # Override PD gains + force limit on every active joint. The defaults in
    # HandKinematicModel.__init__ (kp=400, kd=10) are underdamped for v6's
    # 5-finger packing — joints oscillate when target jumps and self-collision
    # pushes them around. kd=40 with kp=400 gives a near-critically-damped
    # response.
    for joint in hand.all_joints:
        joint.set_drive_property(args.kp, args.kd, force_limit=args.force_limit)

    if args.no_self_collision:
        disable_self_collision(hand.hand)
        print('[replay] self-collision disabled (intra-articulation contacts filtered)')

    viewer_env = hand.get_viewer_env()

    # Run!
    while True:
        if args.kinematic:
            # Render-only: skip scene.step() so PhysX won't fight self-collision
            # contacts. The on-screen pose is then exactly what the IK predicted.
            viewer_env.scene.update_render()
            viewer_env.viewer.render()
        else:
            for i in range(10):
                viewer_env.update()

        result = mocap.get()

        if result['status'] == 'recording' and result["result"] is not None:
            points = result["result"]
            qpos = model.forward(points)

            if args.kinematic:
                # Snap qpos directly (no PD drive, no physics) — mirrors the
                # --kinematic path in glove_based/geort/env/hand_debug.py.
                lo, hi = hand.get_joint_limit()
                qpos_clipped = np.clip(qpos,
                                       np.asarray(lo) + 1e-3,
                                       np.asarray(hi) - 1e-3)
                qpos_sim = hand.convert_user_order_to_sim_order(qpos_clipped)
                hand.hand.set_qpos(qpos_sim)
                hand.hand.set_qvel(np.zeros_like(qpos_sim))
            else:
                hand.set_qpos_target(qpos)

        if result['status'] == 'quit':
            break

if __name__ == '__main__':
    main()
