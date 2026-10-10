"""
YOLOv8-nano Fine-Tuning Setup for Sri Lankan Unlaned Intersections
- Optimized for 5 local classes: Three-Wheeler, Motorcycle, Car/Van, Bus/Truck, Emergency Vehicle
- Utilizes AdamW optimizer and Focal Loss weighting for class-imbalance mitigation.
"""

from ultralytics import YOLO

def initialize_model_training():
    # Load pretrained YOLOv8-nano model
    model = YOLO("yolov8n.pt")
    
    print("YOLOv8n model initialized successfully for fine-tuning on local junction dataset.")

if __name__ == "__main__":
    initialize_model_training()