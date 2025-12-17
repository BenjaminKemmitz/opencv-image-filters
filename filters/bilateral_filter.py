# Bilateral Filter Smooths images while preserving edges by combining spatial and intensity information—ideal for denoising without blurring edges.

import cv2

def bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
    filtered = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    return filtered
