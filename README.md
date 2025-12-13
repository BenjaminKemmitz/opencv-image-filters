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
Applies a Gaussian kernel to smooth an image, reducing noise and detail while preserving overall structure.
2. Median Filter
Replaces each pixel with the median of its neighborhood, highly effective for removing salt-and-pepper noise.
3. Bilateral Filter
Smooths images while preserving edges by combining spatial and intensity information—ideal for denoising without blurring edges.
4. Sobel Operator
Computes gradient magnitude in the x and y directions to highlight vertical and horizontal edges.
5. Laplacian Filter
Uses the second derivative of pixel intensity to detect regions of rapid intensity change—useful for highlighting fine edges.
6. Canny Edge Detector
A multi-stage edge detection algorithm that uses gradients, smoothing, and thresholding to extract clean, accurate edges.
7. Morphological Operations
Includes erosion, dilation, opening, and closing to manipulate image structures—often used for noise removal and shape analysis.
8. HSV Color Segmentation
Converts the image to HSV space and isolates pixels within color ranges, allowing robust color-based masking and detection.
9. Basic Thresholding
Converts an image to binary by comparing pixel intensity to a fixed threshold value.
10. Adaptive Thresholding
Determines local thresholds for small regions, performing well in uneven lighting conditions.
11. Otsu’s Thresholding
Automatically computes the optimal threshold by maximizing inter-class variance—ideal when the image histogram is bimodal.
12. Harris Corner Detector
Detects corners by analyzing variations in intensity in local windows—useful for tracking and alignment tasks.
13. Shi-Tomasi Good Features to Track
An improvement over Harris that selects the most stable corners for tracking in real-time applications.
14. ORB Feature Detector
A fast, rotation-invariant keypoint detector and descriptor generator used in SLAM, mapping, and object recognition.
15. FAST Corner Detector
Extremely fast corner detector ideal for real-time vision tasks, especially in embedded or robotics systems.
16. Custom Sharpening Filter
Enhances edges and fine details using a manually designed convolution kernel to increase image crispness.
# Output (Will Add Screenshots)

assets/examples/blur_result.png
assets/examples/cartoon_result.png

# Extending the Project

Next steps:
Add real-time webcam filter versions.
Add custom convolution kernels (motion blur, Sobel, Laplacian).
Implement CUDA acceleration using OpenCV GPU.
Package as a pip-installable module.

# Contributing

This repository is part of a pre-college robotics/computer-vision learning portfolio.
Future pull requests may add more filters or real-time versions.

# License

MIT License (or whichever you prefer)
