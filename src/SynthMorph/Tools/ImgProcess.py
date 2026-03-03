# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:44:26 2025

@author: Administrator
"""

import os
import cv2
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import xlwt


def Picture_Load(image_path):

    if image_path == None:
        image_path = input("Please enter the image path: ").strip()

    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist")
        return None

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image '{image_path}', please check the file format")
        return None

    # Get image size
    height, width = image.shape[:2]
    print(f"Image size: {width}×{height} pixels")

    # Check if image size is 100x100, if not, automatically resize
    if width != 100 or height != 100:
        print(f"Image size is not 100×100 pixels, but {width}×{height} pixels, automatically resizing to 100×100 pixels")
        # Resize to 100x100 using bilinear interpolation
        image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
        print("✓ Automatically resized to 100×100 pixels")
    else:
        print("✓ Image size meets requirements (100×100 pixels)")

    # Convert to single-channel grayscale image
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("✓ Converted to single-channel grayscale image")
    else:
        gray_image = image
        print("✓ Image is already single-channel")

    # Check the range of the grayscale image and convert to a 0-1 density matrix
    # Normalize according to the brightness range of the input image
    min_val = np.min(gray_image)
    max_val = np.max(gray_image)

    if max_val > min_val:
        # Normalize to 0-1 range
        density_matrix = (gray_image.astype(np.float32) -
                          min_val) / (max_val - min_val)
        print(f"✓ Normalized to 0-1 range (original range: {min_val}-{max_val})")
    else:
        # If all pixel values are the same, create an all-0 or all-1 matrix
        density_matrix = np.zeros((100, 100), dtype=np.float32)
        if min_val > 0:
            density_matrix[:, :] = 1.0
        print("✓ Created constant value density matrix")
    return density_matrix

def from_image_to_contour(image_path, contour_path):
    
    
    density_data= Picture_Load(image_path)
    # Resize to target size (width, height)
    density_data = cv2.resize(
        density_data.astype(np.float32),  # Ensure data type is float32
        (1000, 1000),
        interpolation=cv2.INTER_LINEAR)

    # Create grayscale image (inverted: 1 means material, 0 means void)
    gray_image = density_data

    # Create 3×3 tiled image
    density_data_9 = np.tile(gray_image, (3, 3))

    # plt.figure()
    # plt.imshow(density_data_9, cmap='gray')

    # Apply stronger blur (increase sigma value)
    smoothed_density = gaussian_filter(density_data_9, sigma=10.0, mode='wrap')  # Increase sigma value

    # plt.figure()
    # plt.imshow(smoothed_density, cmap='gray')

    # Crop the central region
    cut_density = smoothed_density[990:2010, 990:2010]

    plt.figure()
    plt.imshow(cut_density, cmap='gray')

    # Convert to 8-bit image
    cut_density_8bit = (cut_density * 255).astype(np.uint8)
    _, binary_image1 = cv2.threshold(cut_density_8bit, 127, 255, cv2.THRESH_BINARY)

    plt.figure()
    plt.imshow(binary_image1, cmap='gray')

    # Apply morphological operations to smooth boundaries (closing fills small holes, opening removes small noise)
    kernel = np.ones((3, 3), np.uint8)
    smoothed_binary = cv2.morphologyEx(binary_image1, cv2.MORPH_CLOSE, kernel)
    smoothed_binary = cv2.morphologyEx(smoothed_binary, cv2.MORPH_OPEN, kernel)

    # Find contours (detect only the outermost contours)
    contours, hierarchy = cv2.findContours(
        smoothed_binary,
        cv2.RETR_EXTERNAL,  # Only detect outer contours
        cv2.CHAIN_APPROX_NONE)
    output_path = os.path.join(contour_path,"test.txt")
    f=open(output_path,'w')
    
    for i in range(len(contours)):
        contour_data=contours[i].squeeze(axis=1).tolist()

        for j in range(len(contour_data)):
            x, y = contour_data[j][0], contour_data[j][1]
            f.write(str(x/(1000.0/30.0))+'\t')
        f.write('\n')
        # Write y coordinates
        for j in range(len(contour_data)):
            x, y = contour_data[j][0], contour_data[j][1]
            f.write(str(y/(1000.0/30.0))+'\t')
        f.write('\n')
    
    f.close()

    print('Contour saved in '+ output_path)



if __name__=='__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python LLM_CAE_FE_1.py <image_path> <contour_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    contour_path = sys.argv[2]
    from_image_to_contour(image_path, contour_path)
