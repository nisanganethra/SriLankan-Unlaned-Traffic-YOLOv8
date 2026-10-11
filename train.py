from ultralytics import YOLO

# Load a pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Start training
results = model.train(data="data.yaml", epochs=50, imgsz=640)