import argparse
import numpy as np

from geort.replay_mocap import ReplayMocap
from geort.env.hand import HandKinematicModel
from geort.env.hand_static import color_links
from geort import load_model, get_config


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
