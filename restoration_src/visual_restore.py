import cv2
import matplotlib.pyplot as plt

def save_before_after(original_bgr, processed_bgr, out_path):
    orig = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    proc = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.imshow(orig); plt.title('Before'); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(proc); plt.title('After'); plt.axis('off')
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def save_histograms(original_bgr, processed_bgr, out_path):
    orig = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    proc = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(8, 4))
    plt.hist(orig.ravel(), bins=256, range=(0, 255), alpha=0.6, label='Before')
    plt.hist(proc.ravel(), bins=256, range=(0, 255), alpha=0.6, label='After')
    plt.title('Gray-level Histogram'); plt.xlabel('Intensity'); plt.ylabel('Count'); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def save_edges_preview(edges_img, out_path, title='Edges'):
    plt.figure(figsize=(5, 5))
    plt.imshow(edges_img, cmap='gray')
    plt.title(title); plt.axis('off')
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
