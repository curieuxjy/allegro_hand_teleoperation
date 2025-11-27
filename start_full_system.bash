#!/bin/bash
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

# Full system startup script for Manus glove teleoperation with GeoRT
# This script builds the ROS2 workspace and launches all required nodes

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default checkpoint values (can be overridden)
RIGHT_CKPT="${RIGHT_CKPT:-miller_right_1120_104808_allegro_right_2025-11-20_12-29-03_s10}"
LEFT_CKPT="${LEFT_CKPT:-miller_left_1120_103637_allegro_left_2025-11-20_12-29-12_s10}"
LOOP_HZ="${LOOP_HZ:-100.0}"
SMOOTHING_ALPHA="${SMOOTHING_ALPHA:-0.9}"

echo -e "${GREEN}=== Allegro Hand Teleoperation Full System Startup ===${NC}"
echo ""

# Step 1: Build ROS2 workspace
echo -e "${YELLOW}[1/2] Building ROS2 workspace...${NC}"
cd glove_based/geort_ws
colcon build --symlink-install
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build completed successfully${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Source the workspace
source install/setup.bash
cd ../..

echo ""
echo -e "${YELLOW}[2/2] Launching full system with ROS2 launch...${NC}"
echo -e "  Right checkpoint: ${GREEN}${RIGHT_CKPT}${NC}"
echo -e "  Left checkpoint:  ${GREEN}${LEFT_CKPT}${NC}"
echo -e "  Loop Hz:          ${GREEN}${LOOP_HZ}${NC}"
echo -e "  Smoothing alpha:  ${GREEN}${SMOOTHING_ALPHA}${NC}"
echo ""

# Step 2: Launch all nodes using ROS2 launch
cd glove_based/geort_ws
ros2 launch launch/full_system_bringup.launch.py \
    right_ckpt:="${RIGHT_CKPT}" \
    left_ckpt:="${LEFT_CKPT}" \
    loop_hz:="${LOOP_HZ}" \
    smoothing_alpha:="${SMOOTHING_ALPHA}"
