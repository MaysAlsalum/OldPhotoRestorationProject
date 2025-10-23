import cv2
import numpy as np

def median_denoise(img, ksize=3):
    k = max(3, int(ksize) | 1)  # force odd
    return cv2.medianBlur(img, k)

def gaussian_denoise(img, ksize=5, sigma=0):
    k = max(3, int(ksize) | 1)  # force odd
    return cv2.GaussianBlur(img, (k, k), sigma)
