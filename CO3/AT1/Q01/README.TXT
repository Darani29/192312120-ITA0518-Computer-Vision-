import cv2
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
import os

# 1. Trigger the direct browser upload prompt
print("Please select and upload your image file:")
uploaded = files.upload()

# Get the filename of the uploaded file
if uploaded:
    image_path = list(uploaded.keys())[0]
    
    # 2. Load and prepare input image as Grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 3. Set threshold levels (low, high)
    thresh1 = (50, 100)
    thresh2 = (100, 200)
    thresh3 = (150, 250)

    # 4. Apply Canny at each level
    edges1 = cv2.Canny(img_blur, thresh1[0], thresh1[1])
    edges2 = cv2.Canny(img_blur, thresh2[0], thresh2[1])
    edges3 = cv2.Canny(img_blur, thresh3[0], thresh3[1])

    # 5. Plotting results side-by-side
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Input')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(edges1, cmap='gray')
    plt.title('Low Thresh: 50, 100')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(edges2, cmap='gray')
    plt.title('Ideal Thresh: 100, 200')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(edges3, cmap='gray')
    plt.title('High Thresh: 150, 250')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("No file uploaded. Please re-run the cell and select a file.")

