# Real-Time Traffic Volume Estimation and Emergency Vehicle Priority at Unlaned Intersections in Sri Lanka

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
```
---

## Dataset & Curation Protocol

* **Primary Collection Protocol**: 3-hour stratified field video capture ($1080\text{p at } 30\text{ FPS}$) from elevated vantage points ($35^\circ$ downward pitch) across major unlaned Colombo junctions (Rajagiriya, Borella, Battaramulla).
* **Sample Size**: 4,000 curated, anonymized images extracted at $1\text{ FPS}$ with over 48,000+ individual labeled vehicle instances.
* **Target Classes**: Three-Wheeler (35%), Motorcycle (30%), Car_Van (20%), Bus_Truck (10%), and Emergency_Vehicle (5%).
* **Ethical Compliance**: Automated Gaussian blurring ($21 \times 21$ kernel) applied across detected pedestrian faces and vehicle license plates prior to annotation.

---

## Quick Start & Setup
### Prerequisites
Ensure Python 3.10+ and Git are installed on your local environment or edge device.

### Clone the Repository
```bash
git clone [https://github.com/NethraDiwaman/SriLankan-Unlaned-Traffic-YOLOv8.git](https://github.com/NethraDiwaman/SriLankan-Unlaned-Traffic-YOLOv8.git)
cd SriLankan-Unlaned-Traffic-YOLOv8
```
### Create a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```
### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running Preprocessing Scripts
### Run Privacy Anonymization
To execute privacy scrubbing on extracted frame folders:
```bash
python scripts/anonymize_privacy.py
```
### Compute Inter-Annotator Agreement (IAA)
To validate bounding-box overlap consistency between labeling passes:
```bash
python scripts/calculate_iaa.py
```

---

## Experimental Framework & Methodology Pipeline

```mermaid
flowchart TD
    %% Input Stage
    A["Primary Field Video Dataset <br/> 3-Hour Capture - 1080p @ 30 FPS"] --> B["Data Preprocessing Pipeline <br/> Resizing 640x640 - CLAHE - Privacy Blur"]

    %% Feature Extraction & Detection Module
    B --> C["Model Architecture Fine-Tuning <br/> YOLOv8-nano - Decoupled Head"]

    %% Spatial Tracking Module
    C --> D["Spatial Multi-Object Tracking <br/> ByteTRACK State Estimation"]

    %% Computational Analysis Modules
    D --> E["Density Analysis Engine <br/> Area Occupancy Ratio Calculation - AOR"]
    D --> F["Emergency Preemption Module <br/> Class Identification & Priority Logic"]

    %% Experimental Evaluation & Benchmarking
    E --> G["Performance Evaluation & Benchmarking <br/> Comparison vs. Fixed-Timer Baseline"]
    F --> G

    %% Styling
    style A fill:#2d3748,stroke:#4a5568,color:#fff
    style B fill:#2d3748,stroke:#4a5568,color:#fff
    style C fill:#2b6cb0,stroke:#3182ce,color:#fff
    style D fill:#2b6cb0,stroke:#3182ce,color:#fff
    style E fill:#2f855a,stroke:#38a169,color:#fff
    style F fill:#c53030,stroke:#e53e3e,color:#fff
    style G fill:#d69e2e,stroke:#ecc94b,color:#fff
```

---

## Authors & Academic Context
* W.G.N. Nethra Diwaman (Student ID: ITBIN-2312-0013)
* Rasanga V. Bandara (Student ID: ITBIN-2312-0002)

* Module: IT41043 — Intelligent Systems
* Faculty: Faculty of Information Technology, Horizon Campus, Sri Lanka.
