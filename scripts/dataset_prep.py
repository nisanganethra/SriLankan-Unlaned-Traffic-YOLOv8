import os
import cv2
import numpy as np
import random
from anonymize_privacy import blur_faces_and_plates

RAW_DIR = "data/raw"
PROCESSED_IMG_DIR = "data/processed/images"
PROCESSED_LBL_DIR = "data/processed/labels"

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(PROCESSED_IMG_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_LBL_DIR, split), exist_ok=True)

def apply_preprocessing_pipeline(image):
    # 1. Privacy Anonymization (Faces & Plates)
    anonymized = blur_faces_and_plates(image)

    # 2. CLAHE Contrast Enhancement on Luminance Channel (Lab Color Space)
    lab = cv2.cvtColor(anonymized, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_Lab2BGR)

    # 3. Resolution Standardization (640x640 bilinear interpolation)
    resized = cv2.resize(enhanced, (640, 640), interpolation=cv2.INTER_LINEAR)
    return resized

def process_junction_footages():
    frame_counter = 0
    junctions = ['koswatta', 'cinec']
    
    for junction in junctions:
        path = os.path.join(RAW_DIR, junction)
        if not os.path.exists(path):
            continue
        videos = [v for v in os.listdir(path) if v.endswith(('.mp4', '.avi', '.mov'))]
        for vid in videos:
            cap = cv2.VideoCapture(os.path.join(path, vid))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            interval = int(fps) # Extract 1 frame per second (1 FPS)
            success, frame = cap.read()
            
            while success:
                if frame_counter % interval == 0:
                    processed = apply_preprocessing_pipeline(frame)
                    r = random.random()
                    split = 'train' if r < 0.70 else ('val' if r < 0.85 else 'test')
                    filename = f"{junction}_frame_{frame_counter:06d}.jpg"
                    cv2.imwrite(os.path.join(PROCESSED_IMG_DIR, split, filename), processed)
                
                success, frame = cap.read()
                frame_counter += 1
            cap.release()
    print("Preprocessing pipeline completed successfully for Koswatta and Cinec footages.")

if __name__ == "__main__":
    process_junction_footages()