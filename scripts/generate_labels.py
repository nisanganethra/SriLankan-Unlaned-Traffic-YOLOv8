import os
from pathlib import Path
from ultralytics import YOLO

# Load a pre-trained YOLO model to bootstrap pseudo-labels
model = YOLO("yolov8x.pt")

splits = ["train", "val"]
img_base_dir = Path("data/processed/images")
lbl_base_dir = Path("data/processed/labels")

for split in splits:
    img_dir = img_base_dir / split
    lbl_dir = lbl_base_dir / split
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(img_dir.glob("*.jpg"))
    print(f"Generating pseudo-labels for {split} split ({len(image_paths)} images)...")
    
    for img_path in image_paths:
        results = model(img_path, verbose=False, conf=0.3)[0]
        txt_path = lbl_dir / f"{img_path.stem}.txt"
        
        with open(txt_path, "w") as f:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                xywhn = box.xywhn[0].tolist() # Normalized [x_center, y_center, width, height]
                
                # Map COCO vehicle classes to your 4 custom classes:
                # COCO: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck -> map to car index 0)
                if cls_id == 2 or cls_id == 7:
                    mapped_cls = 0  # car
                elif cls_id == 5:
                    mapped_cls = 1  # bus
                elif cls_id == 3:
                    mapped_cls = 2  # motorcycle
                else:
                    continue
                
                f.write(f"{mapped_cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")

print("Label generation complete! You can now run train.py.")