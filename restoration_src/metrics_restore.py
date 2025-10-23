import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def compute_psnr(ref, test):
    return psnr(ref, test, data_range=255)

def compute_ssim(ref, test):
    ref_y = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    test_y = cv2.cvtColor(test, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    return ssim(ref_y, test_y, data_range=255)
