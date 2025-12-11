# OpenCV Image Filters Collection

A collection of classic and custom image-processing filters implemented using Python + OpenCV.
This repository is part of a larger computer-vision learning portfolio demonstrating core image-processing fundamentals.

# Project Goals

Build a reusable set of OpenCV-based filters.

Learn convolution, kernels, and image transformations.

Showcase clean, documented Python code suitable for a robotics/computer-vision portfolio.

Provide a modular folder structure usable in future robotics/AI projects.

# Requirements

Install dependencies:

pip install opencv-python numpy

# How to Run

Run any filter individually:

python filters/blur_filter.py --image assets/sample.jpg


Or run the main demo:

python main.py --image assets/sample.jpg --filter blur

# Available Filters
1. Gaussian Blur
  Demonstrates smoothing and noise reduction.
2. Median Filter
3. Bilateral Filter
4. Sobel Operator
5. Laplacian Filter
6. Canny Edge Detector
7. Morphological Operations
9. HSV Color Segmentation
10. Thresholding
11. Adaptive Thresholding
12. Otsu’s Thresholding
13. Harris Corner Detector
14. Shi-Tomasi Good Features to Track
15. ORB Feature Detector
16. FAST Corner Detector
17. Custom Sharpening Filter

# Example Output (Optional Screenshots Section)

You may later add:

assets/examples/blur_result.png
assets/examples/cartoon_result.png

# Extending the Project

Good next steps:

Add real-time webcam filter versions.

Add custom convolution kernels (motion blur, Sobel, Laplacian).

Implement CUDA acceleration using OpenCV GPU.

Package as a pip-installable module.

# Contributing

This repository is part of a pre-college robotics/computer-vision learning portfolio.
Future pull requests may add more filters or real-time versions.

# License

MIT License (or whichever you prefer)
