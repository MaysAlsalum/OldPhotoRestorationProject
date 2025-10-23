import cv2
import numpy as np

def sobel_edges(gray, ksize=3):
    k = max(1, int(ksize))
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    mag = cv2.magnitude(gx, gy)
    mag = (mag / (mag.max() + 1e-8) * 255).astype('uint8')
    return mag

def canny_edges(gray, t1=100, t2=200):
    return cv2.Canny(gray, int(t1), int(t2))