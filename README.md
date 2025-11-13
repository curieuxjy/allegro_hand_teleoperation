# allegro_hand_teleoperation

Teleoperation Application Code for the Allegro Hand Platforms

This repository provides a teleoperation framework for the Allegro Hand (V4 series).  
The core feature is **hand retargeting**, which converts glove-based or vision-based inputs into Allegro Hand joint commands.

> ⚠ **Note:**  
> This project provides **Hand Teleoperation code only**.  
> **Robot Arm Teleoperation must be developed separately.**

---

## Dependency: Allegro Hand ROS2 Controller

This project operates based on the official controller below:

👉 https://github.com/Wonikrobotics-git/allegro_hand_ros2

---

## Contents Overview

### **1. Glove-Based Teleoperation (Manus Glove)**

#### **1-1. Rule-Based Retargeting**
Simple mapping approach for joint transformation.

#### **1-2. GeoRT-Inspired Retargeting**
AI-based mapping inspired by the GeoRT architecture.  
Reference: https://zhaohengyin.github.io/geort/#

---

### **2. Vision-Based Teleoperation**

Camera-based hand pose estimation → retargeting to Allegro Hand.  
Reference: https://github.com/dexsuite/dex-retargeting

---

## Third-Party Hardware (Optional)

- **Manus Meta Glove**  
  Documentation: https://docs.manus-meta.com/3.0.0/Plugins/SDK/ROS2/getting%20started/  
  (*Manus SDK requires separate installation*)

---

## Referenced Research

- **GeoRT**  
  https://zhaohengyin.github.io/geort/#  
  (CC-BY-NC / Code not included)

- **DexSuite – Dex-Retargeting**  
  https://github.com/dexsuite/dex-retargeting  
  (Reference for concepts)

---

## Future Expansion

- Expansion to additional Allegro Hand platforms  
- Additional retargeting methods (using different hardware or algorithms)

---

## License

MIT License

- Modifications © 2025 **WonikRobotics_official**  
- Additional contributions © 2025 **Jungyeon Lee**

See the full license in the [LICENSE](./LICENSE) file.
