import numpy as np

def compute_iou(boxA, boxB):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    box = [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

if __name__ == "__main__":
    b1 = [50, 50, 150, 150]
    b2 = [60, 60, 150, 150]
    print(f"Sample Inter-Annotator IoU Score: {compute_iou(b1, b2):.4f}")