# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project Overview

**allegro_hand_teleoperation** is a ROS2-based teleoperation framework for the Wonik Robotics Allegro Hand (V4 series). The core feature is **hand retargeting** — converting human hand inputs (glove or vision) into Allegro 16-DOF joint commands. This repo contains **hand teleoperation only**; arm teleoperation is out of scope.

### Stack & Dependencies

- **OS / ROS:** Ubuntu 22.04, ROS2 Humble
- **Hardware:** Allegro Hand V4 (single or dual), Manus Metagloves (Manus Core 3 SDK v3.0.0)
- **Upstream controller:** [`Wonikrobotics-git/allegro_hand_ros2`](https://github.com/WonikRobotics-git/allegro_hand_ros2) — must be built and sourced separately
- **Sim:** Sapien 2.2.2 (for GeoRT eval)
- **Python:** 3.10+, conda env `geort` for the learning-based path

### Repository Layout

```
allegro_hand_teleoperation/
├── README.md                  # Top-level overview + setup
├── GLOVE_BASED.md             # Detailed docs for glove path (both methods)
├── start_one_hand.bash        # CAN0 bring-up (single hand)
├── start_two_hands.bash       # CAN0 + CAN1 bring-up (dual hands)
├── glove_based/
│   ├── allegro.py             # AllegroCommandForwarder: activates controller + publishes 16-D Float64MultiArray
│   ├── rule_based_retargeting.py   # Method 1: heuristic glove→Allegro mapping with Open3D viz
│   ├── manus_mocap.py         # PoseArray subscriber for Manus poses (configurable topic/node name)
│   ├── manus_skeleton_21.py   # Preprocessor: glove → 21-joint skeleton (publishes /manus_poses[_l|_r])
│   ├── common_viz.py          # Shared Open3D viz + coord frame conversions
│   ├── geort_data_logger.py   # Method 2 step 1: log human hand .npy
│   ├── geort_realtime_evaluation.py / geort_replay_evaluation.py  # Sapien sim eval
│   ├── geort_allegro_deploy.py        # Dual-hand hardware deployment
│   ├── geort_allegro_deploy_single.py # Single-hand hardware deployment
│   ├── geort/                 # GeoRT package: model, trainer, dataset, loss, generate_robot_data
│   │   ├── config/            # allegro_left.json, allegro_right.json (URDF + fingertip mapping)
│   │   └── env/               # Sapien hand env
│   ├── assets/                # URDFs + meshes for allegro_left, allegro_right
│   ├── data/                  # .npy human logs, .npz robot datasets, chamfer .html viz
│   └── checkpoint/            # Trained GeoRT models (best.pth / last.pth / config.json)
└── materials/                 # README images/GIFs
```

### Two Retargeting Methods

1. **Rule-Based** (`rule_based_retargeting.py`): heuristic 20-D glove ergonomics → 16-D Allegro angles. Pipeline: extract → degree mapping → radians → per-joint scale/offset → clip to `ALLEGRO_LOWER_LIMITS`/`ALLEGRO_UPPER_LIMITS` → EMA smoothing. No training. Immediate deployment.
2. **GeoRT** (`glove_based/geort/`): learning-based geometric retargeting (inspired by https://zhaohengyin.github.io/geort/). Workflow: log human data → generate 1M robot kinematic samples → train (chamfer + curvature + pinch losses, wandb logging) → eval in Sapien → deploy to hardware. `best.pth` (lowest training loss) is the default deployment checkpoint.

### ROS2 Topic Conventions (matters when editing)

- **Single hand:** `/allegro_hand_position_controller/commands` (`Float64MultiArray`, 16 floats)
- **Dual hands:** `/allegro_hand_position_controller_r/commands` and `..._l/commands`
- **Manus gloves:** scripts subscribe to `manus_glove_left` / `manus_glove_right` — the Manus ROS2 publisher must be **hardcoded** with the user's actual glove IDs (see GLOVE_BASED.md "Hardcode Left/Right Glove Recognition"). Without this, scripts silently receive no data.
- **Preprocessed skeleton:** `/manus_poses` (single) or `/manus_poses_left` / `/manus_poses_right` (dual)
- Controller activation happens automatically inside `AllegroCommandForwarder` via the `controller_manager/switch_controller` service.

### Allegro Joint Order (hardware)

16-D vector ordered as: **Thumb (4) → Index (4) → Middle (4) → Ring (4)**. GeoRT model output is in (Index, Middle, Ring, Thumb) order and is reordered in `post_processing_commands()`. Don't reorder without checking both ends.

### Safety Behaviors

- All deployers return to a neutral `base_position` on shutdown (`AllegroCommandForwarder.return_to_base`).
- EMA smoothing (`smoothing_alpha`) and per-tick rate limiting (`max_delta`) are applied before publishing — keep them in place when modifying control loops.
- Joint outputs are clipped to hardware limits before publishing.

### When Working in This Repo

- **Don't touch upstream `allegro_hand_ros2`** from here — it's a separate workspace the user builds and sources.
- **Rule-based vs GeoRT are independent paths** — changes in one shouldn't affect the other unless explicitly asked.
- **`AllegroCommandForwarder` is shared** between rule-based and GeoRT deployers; changes here ripple to every entry point.
- Single-hand vs dual-hand is controlled by the `two_robots` flag in `AllegroCommandForwarder`, not by separate code paths — preserve this when refactoring.
- The repo intentionally has separate `geort_allegro_deploy.py` and `geort_allegro_deploy_single.py` scripts that share the `GeortAllegroDeployer` class. Don't collapse them without asking.
- Vision-based teleop is referenced in README (`VISION_BASED.md`) but not yet present — don't assume its code exists.
