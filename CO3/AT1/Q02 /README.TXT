import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# -------------------------------------------------------------------------
# 1. Browse & Upload Image from Folder (Google Colab File Picker)
# -------------------------------------------------------------------------
print("Click 'Choose Files' below to select an image from your folder:")
uploaded = files.upload()

# Retrieve the filename of the uploaded image
filename = list(uploaded.keys())[0]

# Read the uploaded image in grayscale mode (0 flag)
img_noisy = cv2.imread(filename, 0)

# -------------------------------------------------------------------------
# 2. Apply Noise Reduction Filters
# -------------------------------------------------------------------------

# A. Median Filter (Kernel size = 5)
median_filtered = cv2.medianBlur(img_noisy, ksize=5)

# B. Gaussian Filter (Kernel size = 5x5, SigmaX = 0)
gaussian_filtered = cv2.GaussianBlur(img_noisy, ksize=(5, 5), sigmaX=0)

# C. Bilateral Filter (d = 9, sigmaColor = 75, sigmaSpace = 75)
bilateral_filtered = cv2.bilateralFilter(img_noisy, d=9, sigmaColor=75, sigmaSpace=75)

# D. Non-Local Means (NLM) Denoising (h = 10, templateWindowSize = 7, searchWindowSize = 21)
nlm_filtered = cv2.fastNlMeansDenoising(img_noisy, None, h=10, templateWindowSize=7, searchWindowSize=21)

# -------------------------------------------------------------------------
# 3. Visual Comparison Output
# -------------------------------------------------------------------------
plt.figure(figsize=(18, 10))

# Subplot 1: Uploaded Input Image
plt.subplot(2, 3, 1)
plt.imshow(img_noisy, cmap='gray')
plt.title(f'1. Uploaded Image\n({filename})', fontsize=12, fontweight='bold')
plt.axis('off')

# Subplot 2: Median Filter Output
plt.subplot(2, 3, 2)
plt.imshow(median_filtered, cmap='gray')
plt.title('2. Median Filter\n(Best for Impulse / Salt-and-Pepper)', fontsize=12, fontweight='bold')
plt.axis('off')

# Subplot 3: Gaussian Filter Output
plt.subplot(2, 3, 3)
plt.imshow(gaussian_filtered, cmap='gray')
plt.title('3. Gaussian Filter\n(Smooths Noise, Blurs Edges)', fontsize=12, fontweight='bold')
plt.axis('off')

# Subplot 4: Bilateral Filter Output
plt.subplot(2, 3, 4)
plt.imshow(bilateral_filtered, cmap='gray')
plt.title('4. Bilateral Filter\n(Preserves Sharp Edges)', fontsize=12, fontweight='bold')
plt.axis('off')

# Subplot 5: Non-Local Means Output
plt.subplot(2, 3, 5)
plt.imshow(nlm_filtered, cmap='gray')
plt.title('5. NLM Denoising\n(Preserves Complex Textures)', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()
