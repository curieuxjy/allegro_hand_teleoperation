# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Teleoperation framework for Allegro Hand V4 robotic hands using Manus gloves. Core feature is **hand retargeting** - converting glove-based inputs into Allegro Hand joint commands via ROS2.

**Dependencies:**
- ROS2 Humble (Ubuntu 22.04)
- [allegro_hand_ros2](https://github.com/WonikRobotics-git/allegro_hand_ros2) controller
- MANUS Core 3 SDK v3.0.0 with ROS2 package
- Python 3.10+

## Common Commands

### Environment Setup

```bash
# GeoRT conda environment
conda create -n geort python=3.10
conda activate geort
pip install -r glove_based/geort_requirements.txt
cd glove_based && pip install -e .
```

### Running Teleoperation

**Rule-Based (heuristic):**
```bash
# Single robot, right hand
python glove_based/rule_based_retargeting.py --setup single

# Dual robots, both hands
python glove_based/rule_based_retargeting.py --setup dual
```

**GeoRT Learning-Based:**
```bash
# Log human hand data (requires manus_skeleton_21.py running)
python glove_based/geort_data_logger.py --name human1 --handness right --duration 60 --hz 30

# Generate robot kinematics data
python glove_based/geort/generate_robot_data.py --hand allegro_right

# Train model
python glove_based/geort/trainer.py --hand allegro_right --human_data human1_right_*.npy

# Deploy to hardware (dual hands)
python glove_based/geort_allegro_deploy.py --right_ckpt "CKPT_TAG" --left_ckpt "CKPT_TAG"

# Deploy single hand
python glove_based/geort_allegro_deploy_single.py --ckpt "CKPT_TAG" --side right
```

### ROS2 Launch Commands

```bash
# Start Manus data publisher
ros2 run manus_ros2 manus_data_publisher

# Preprocess glove data to 21-joint skeleton
python glove_based/manus_skeleton_21.py

# Launch Allegro controller (single)
ros2 launch allegro_hand_bringup allegro_hand.launch.py

# Launch Allegro controller (dual)
ros2 launch allegro_hand_bringup allegro_hand_duo.launch.py
```

## Architecture

### Directory Structure

```
glove_based/
├── rule_based_retargeting.py       # Heuristic glove-to-allegro mapping
├── allegro.py                      # AllegroCommandForwarder ROS2 node
├── manus_mocap.py                  # Manus glove mocap subscriber
├── manus_skeleton_21.py            # Glove data preprocessor (21-joint format + pinch detection)
├── geort_allegro_deploy.py         # Dual hand GeoRT deployment
├── geort_allegro_deploy_single.py  # Single hand GeoRT deployment
├── geort_allegro_deploy_pinch.py   # Dual hand deployment with pinch override
├── geort_allegro_deploy_single_pinch.py  # Single hand deployment with pinch override
├── geort_data_logger.py            # Human hand motion recorder
├── geort_replay_evaluation.py      # Offline evaluation in Sapien
├── geort_realtime_evaluation.py    # Live evaluation in Sapien
└── geort/                          # GeoRT learning-based retargeting
    ├── model.py                    # FKModel and IKModel neural networks
    ├── trainer.py                  # Training pipeline with geometric losses
    ├── export.py                   # Model loading (load_model function)
    ├── generate_robot_data.py      # Robot kinematics dataset generator
    └── config/template.py          # Allegro hand configurations
```

### Key Components

**AllegroCommandForwarder** (`allegro.py`): ROS2 node that publishes joint commands to Allegro controllers. Handles controller activation and topic routing:
- Single robot: `/allegro_hand_position_controller/commands`
- Dual robots: `/allegro_hand_position_controller_{r,l}/commands`

**GeoRT Models** (`geort/model.py`):
- `FKModel`: Forward kinematics - joint angles to fingertip positions
- `IKModel`: Inverse kinematics - fingertip positions to joint angles
- Per-finger MLPs with Tanh output normalization

**GeortAllegroDeployer** (`geort_allegro_deploy.py`): Real-time deployment node that:
1. Subscribes to `/manus_poses_{left,right}` for hand poses
2. Applies scale factor from checkpoint config
3. Runs IK model forward pass
4. Post-processes: reorders joints (model order -> hardware order)
5. Applies EMA smoothing and rate limiting
6. Publishes to Allegro controllers

### Data Flow

```
Manus Glove → manus_data_publisher → manus_skeleton_21.py → /manus_poses_{left,right}
                                                                    ↓
                                              GeoRT Model (IK) ← ManusMocap
                                                    ↓
                                            post_processing (reorder joints)
                                                    ↓
                                            EMA smoothing + rate limiting
                                                    ↓
                                    AllegroCommandForwarder → /allegro_hand_position_controller/commands
```

### Joint Ordering

**Model output order:** Index(0:4), Middle(4:8), Ring(8:12), Thumb(12:16)
**Hardware order:** Thumb(0:4), Index(4:8), Middle(8:12), Ring(12:16)

The `post_processing_commands()` function handles this reordering.

### Training Losses

GeoRT trainer uses multiple geometric losses:
- **Chamfer distance** (w_chamfer=80.0): Point cloud alignment
- **Direction loss** (w=1.0): Gradient direction preservation
- **Curvature loss** (w_curvature=0.1): Surface smoothness
- **Pinch loss** (w_pinch=1.0-2.0): Fingertip proximity detection

Checkpoints saved to `glove_based/checkpoint/` with `best.pth` (lowest loss) and `last.pth`.

## ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/manus_glove_left` | ManusGlove | Raw left glove data |
| `/manus_glove_right` | ManusGlove | Raw right glove data |
| `/manus_poses_left` | PoseArray | Preprocessed 21-joint left hand |
| `/manus_poses_right` | PoseArray | Preprocessed 21-joint right hand |
| `/index_pinch_left` | Bool | Index pinch detected (left hand) |
| `/index_pinch_right` | Bool | Index pinch detected (right hand) |
| `/allegro_hand_position_controller/commands` | Float64MultiArray | Single robot commands |
| `/allegro_hand_position_controller_{r,l}/commands` | Float64MultiArray | Dual robot commands |

## Pinch Detection

`manus_skeleton_21.py` publishes `/index_pinch_{left,right}` (Bool) when the distance between pose index 4 (index fingertip) and pose index 8 (thumb tip) is less than `INDEX_PINCH_THRESHOLD` (default: 0.02m).

The `*_pinch.py` deployer variants subscribe to these topics and override thumb/index finger joints to predefined pinch positions when True (Step D in `post_processing_commands()`).

## Important Notes

- Glove topic names must be hardcoded in `ManusDataPublisher.cpp` (line 336) for left/right recognition
- Scale factor from training is automatically loaded from checkpoint's `config.json`
- EMA smoothing (default alpha=0.9) and rate limiting (default max_delta=0.05 rad/tick) are applied during deployment
