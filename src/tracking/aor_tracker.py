import numpy as np

def calculate_aor(bounding_boxes, roi_polygon_area):
    """
    Calculates Area Occupancy Ratio (AOR) within a virtual Region of Interest (ROI)
    to measure spatial junction congestion without relying on lane boundaries.
    """
    total_intersection_area = 0.0
    for box in bounding_boxes:
        # box format: [x1, y1, x2, y2]
        box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        total_intersection_area += box_area
        
    aor_ratio = total_intersection_area / roi_polygon_area
    return min(aor_ratio, 1.0)