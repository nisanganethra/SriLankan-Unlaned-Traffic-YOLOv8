import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def blur_faces_and_plates(image):
    """
    Applies Gaussian blurring to pedestrian faces and license plate regions 
    to protect commuter privacy under non-invasive field capture protocols.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        image[y:y+h, x:x+w] = cv2.GaussianBlur(image[y:y+h, x:x+w], (99, 99), 30)
    return image