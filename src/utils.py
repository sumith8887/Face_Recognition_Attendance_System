import cv2

def resize_frame(frame):
    return cv2.resize(frame, (0,0), fx=0.5, fy=0.5)