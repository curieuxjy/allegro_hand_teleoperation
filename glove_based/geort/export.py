# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import fnmatch
import os
from pathlib import Path

import torch

from geort.formatter import HandFormatter
from geort.model import IKModel
from geort.utils.path import to_package_root, get_checkpoint_root
from geort.utils.config_utils import load_json, parse_config_keypoint_info, parse_config_joint_limit


class GeoRTRetargetingModel:
    '''
        Used by external programs.
    '''
    def __init__(self, model_path, config_path):
        config = load_json(config_path)
        keypoint_info = parse_config_keypoint_info(config)
        joint_lower_limit, joint_upper_limit = parse_config_joint_limit(config)
        self.human_ids = keypoint_info["human_id"]
        self.model = IKModel(keypoint_joints=keypoint_info["joint"]).cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit) # GeoRT will do normalization.

    def forward(self, keypoints):
        # keypoints: [N, 3]
        keypoints = keypoints[self.human_ids] # extract.
        joint_normalized = self.model.forward(torch.from_numpy(keypoints).unsqueeze(0).reshape(1, -1, 3).float().cuda())
        joint_raw = self.qpos_normalizer.unnormalize(joint_normalized.detach().cpu().numpy())
        return joint_raw[0]


def load_model(tag='', epoch=0):
    '''
        Loading API.

        Args:
            tag: Checkpoint matcher. Either a plain substring of the run
                 directory name, or a glob with `*` / `?` (e.g.
                 'v6_right_*_with_collision'). Must match exactly one
                 directory — otherwise we abort with a list of candidates
                 instead of silently using a wrong (or empty) path.
            epoch: Epoch number to load, or special values:
                   - 0 (default): load 'last.pth'
                   - 'best' or -1: load 'best.pth'
                   - positive int: load 'epoch_{epoch}.pth'
    '''
    checkpoint_root = get_checkpoint_root()
    all_entries = os.listdir(checkpoint_root)

    # Only consider directories (skip stray *.pth files like fk_model_*.pth
    # that sit at the checkpoint root)
    candidates = [
        d for d in all_entries
        if os.path.isdir(os.path.join(checkpoint_root, d))
    ]

    # If the tag looks like a glob (contains * or ?), use fnmatch; otherwise
    # fall back to substring match (the historical behavior).
    if tag and ('*' in tag or '?' in tag):
        matches = fnmatch.filter(candidates, tag)
    else:
        matches = [d for d in candidates if tag in d] if tag else []

    if len(matches) == 0:
        hint = "\n  ".join(sorted(candidates)) if candidates else "(none — train a model first)"
        raise FileNotFoundError(
            f"No checkpoint directory matched tag {tag!r} in {checkpoint_root}.\n"
            f"Available run directories:\n  {hint}\n"
            f"Pass any unique substring of one of the above (no need for shell wildcards)."
        )
    if len(matches) > 1:
        hint = "\n  ".join(sorted(matches))
        raise ValueError(
            f"Tag {tag!r} matched {len(matches)} checkpoint directories — disambiguate:\n  {hint}"
        )

    run_dir = Path(checkpoint_root) / matches[0]
    print(f"[load_model] using checkpoint: {run_dir.name}")

    # Handle different epoch specifications
    if epoch == 'best' or epoch == -1:
        model_path = run_dir / "best.pth"
    elif isinstance(epoch, int) and epoch > 0:
        model_path = run_dir / f"epoch_{epoch}.pth"
    else:  # epoch == 0 or default
        model_path = run_dir / "last.pth"

    config_path = run_dir / "config.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weight file not found: {model_path}\n"
            f"Available in this run:\n  " + "\n  ".join(sorted(p.name for p in run_dir.glob('*.pth')))
        )

    return GeoRTRetargetingModel(model_path=model_path, config_path=config_path)

if __name__ == '__main__':
    # load the model in one line.
    load_model(tag="allegro_last")
