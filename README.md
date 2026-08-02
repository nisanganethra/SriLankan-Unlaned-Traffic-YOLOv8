# Real-Time Traffic Volume Estimation and Emergency Vehicle Priority at Unlaned Intersections in Sri Lanka

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-realtime.svg)](https://github.com/ultralytics/ultralytics)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

An edge-based Intelligent Transportation System (ITS) designed to perform localized vehicle detection, density estimation, and emergency vehicle preemption across heterogeneous, unlaned traffic environments in Sri Lanka without relying on active cloud connections.

---

## Project Overview

Urban intersections in developing nations like Sri Lanka operate under severe vehicle heterogeneity (e.g., three-wheelers, motorcycles, buses) and a total absence of lane discipline. Traditional fixed-timer signals and lane-centric Adaptive Traffic Control Systems (ATCS) fail under these conditions, leading to severe delays and blocked emergency medical routes.

This project fine-tunes a single-stage **YOLOv8-nano** model deployed on edge hardware (Raspberry Pi 4B) to:
1. **Detect & Track** heterogeneous vehicles under high inter-vehicle occlusion using **ByteTRACK**.
2. **Compute Real-Time Congestion** using a dynamic **Area Occupancy Ratio (AOR)** within a defined Region of Interest ($ROI$) instead of simple vehicle counts.
3. **Trigger Emergency Preemption** when priority vehicles (`Emergency_Vehicle`) enter the field of view, dynamically clearing traffic paths.

---

## Repository Structure

```text
SriLankan-Unlaned-Traffic-YOLOv8/
├── .gitignore                 # Excludes raw datasets, video streams, and virtual environments
├── README.md                  # Comprehensive project documentation
├── requirements.txt           # Python dependency specifications
├── data/
│   ├── raw/                   # Raw video frame sampling placeholders (.gitkeep)
│   └── processed/             # Anonymized and normalized dataset split structure
├── scripts/
│   ├── anonymize_privacy.py   # Facial & license plate blur preprocessor (Gaussian)
│   ├── calculate_iaa.py       # Inter-annotator agreement validator (IoU & Cohen's Kappa)
│   └── dataset_prep.py        # Frame extraction, resizing (640x640), and split manager
└── src/
    ├── models/                # YOLOv8 fine-tuning configurations & anchor-free head setups
    └── tracking/              # ByteTRACK multi-object integration & Area Occupancy Engine