import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate or Load a Test Pattern Image (Chessboard / Geometric Shapes)
img_clean = np.ones((400, 400), dtype=np.uint8) * 255
# Draw geometric shapes with sharp corners and intersecting lines
cv2.rectangle(img_clean, (80, 80), (200, 200), 50, -1)
cv2.rectangle(img_clean, (150, 150), (320, 320), 100, -1)
cv2.circle(img_clean, (280, 100), 50, 0, -1)

# 2. Function to Run Harris Corner Analysis across Specific Parameters
def run_harris_analysis(image, block_size, ksize, k, thresh_ratio):
    # Convert image to float32 (Required by OpenCV Harris Corner)
    gray = np.float32(image)
    
    # Compute Harris Corner Response Map R
    dst = cv2.cornerHarris(gray, blockSize=block_size, ksize=ksize, k=k)
    
    # Dilate response map for visualization marking
    dst_dilated = cv2.dilate(dst, None)
    
    # Create RGB canvas for marking detected corners
    img_result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Apply threshold to select strong corners
    threshold = thresh_ratio * dst.max()
    corner_mask = dst > threshold
    
    # Mark detected corners as red circles
    img_result[corner_mask] = [255, 0, 0]
    num_corners = np.sum(corner_mask)
    
    return img_result, num_corners

# 3. Parameter Sweep Experiments
params_to_test = [
    # (blockSize, ksize, k, thresh_ratio, Title)
    (2, 3, 0.04, 0.01, "Baseline (block=2, ksize=3, k=0.04, thresh=1%)"),
    (5, 3, 0.04, 0.01, "Large Block Size (block=5)"),
    (2, 7, 0.04, 0.01, "Large Sobel Aperture (ksize=7)"),
    (2, 3, 0.12, 0.01, "High Harris Constant (k=0.12)"),
    (2, 3, 0.04, 0.08, "High Threshold (thresh=8%)")
]

plt.figure(figsize=(16, 10))

for i, (b_size, ks, k_val, t_ratio, title) in enumerate(params_to_test):
    res_img, count = run_harris_analysis(img_clean, b_size, ks, k_val, t_ratio)
    
    plt.subplot(2, 3, i + 1)
    plt.imshow(res_img)
    plt.title(f"{title}\nDetected Corners: {count}", fontsize=10, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.show()
