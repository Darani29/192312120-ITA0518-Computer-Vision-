import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. Load or Generate Test Image with Artificial Illumination Gradient
# -------------------------------------------------------------------------
# Create a base image with distinct shapes
base_img = np.ones((400, 600), dtype=np.uint8) * 128
cv2.circle(base_img, (150, 200), 80, 220, -1)
cv2.rectangle(base_img, (320, 100), (520, 300), 50, -1)

# Generate a strong spatial illumination ramp (bright on left, dark on right)
x = np.linspace(1.8, 0.2, 600)
y = np.linspace(1.8, 0.2, 400)
xx, yy = np.meshgrid(x, y)
illumination_ramp = xx * 0.5

# Apply non-uniform illumination to create an underexposed/overexposed gradient
img_illuminated = np.clip(base_img.astype(np.float32) * illumination_ramp, 0, 255).astype(np.uint8)

# -------------------------------------------------------------------------
# 2. Apply Edge Detection Under Varying Preprocessing Strategies
# -------------------------------------------------------------------------

# Strategy A: Direct Canny on Illumination-Varying Image (Fixed Thresholds)
edges_direct = cv2.Canny(img_illuminated, 50, 150)

# Strategy B: Global Histogram Equalization Preprocessing + Canny
img_he = cv2.equalizeHist(img_illuminated)
edges_he = cv2.Canny(img_he, 50, 150)

# Strategy C: CLAHE (Adaptive Preprocessing) + Canny
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_illuminated)
edges_clahe = cv2.Canny(img_clahe, 50, 150)

# -------------------------------------------------------------------------
# 3. Visualize Inputs, Preprocessed Images, and Detected Edges
# -------------------------------------------------------------------------
plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.imshow(img_illuminated, cmap='gray')
plt.title('1. Non-Uniform Illumination Input', fontsize=10, fontweight='bold')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_he, cmap='gray')
plt.title('2. Global Histogram Equalization', fontsize=10, fontweight='bold')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_clahe, cmap='gray')
plt.title('3. CLAHE Preprocessed', fontsize=10, fontweight='bold')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(edges_direct, cmap='gray')
plt.title('4. Direct Canny (Broken Edges in Shadow)', fontsize=10, fontweight='bold')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(edges_he, cmap='gray')
plt.title('5. Global HE + Canny (Noise Amplified)', fontsize=10, fontweight='bold')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(edges_clahe, cmap='gray')
plt.title('6. CLAHE + Canny (Robust & Continuous)', fontsize=10, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()
