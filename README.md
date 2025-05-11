# YOLO_navigation
## Project Title
AutoGuard: Vision-Based Obstacle Detection and Adaptive Navigation for Autonomous Robots

## Overview
A real-time system leveraging YOLOv10 to detect obstacles (humans, vehicles, humps, etc.) in a robot’s path and generate navigation commands. The solution uses a dynamic safety zone to trigger context-aware responses, enabling safe autonomous movement in unstructured environments.

## Problem Statement                               
Current autonomous robots often:

1. Fail to distinguish between obstacle types (e.g., humans vs static objects)
2. Lack real-time responsiveness in cluttered environments
3. Use rigid path-planning systems that ignore contextual threats
4. Struggle with false positives from complex backgrounds

## Proposed Solution

![](https://i.postimg.cc/fLpyWvmh/Whats-App-Image-2025-05-11-at-12-33-50-54fd15dd.jpg)


## Dataset 
[dataset](https://universe.roboflow.com/dsdg/navi-yrfus)

## Application Use Cases

1. Smart Factories:	AGVs avoiding workers & inventory
2. Agriculture:	Harvesting robots navigating crops
3. Healthcare:	Delivery bots in hospital corridors
4. Smart Cities: Sidewalk cleaning robots

## Conclusion
This project establishes a foundational framework for intelligent robotic navigation, achieving 15 FPS processing with 80%+ accuracy on critical obstacles. The modular architecture allows seamless integration of enhancements like sensor fusion and edge deployment, positioning it as a scalable solution for Industry 4.0 applications
