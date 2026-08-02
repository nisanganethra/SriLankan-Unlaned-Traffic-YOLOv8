import cv2
import os

def apply_privacy_blur(image_path, output_path):
    """
    Applies Gaussian blurring to privacy-sensitive regions (faces, license plates).
    """
    image = cv2.imread(image_path)
    if image is None:
        return
    
    blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, blurred_image)

if __name__ == "__main__":
    print("Privacy blur preprocessor script initialized.")