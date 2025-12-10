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
1. Blur Filter

Uses Gaussian blur.

Demonstrates smoothing and noise reduction.

2. Sharpen Filter

Applies a high-pass convolution kernel.

Enhances edges and details.

3. Edge Detection Filter

Implements Canny Edge Detection.

Great for robotics vision and feature extraction.

4. Emboss Filter

Uses directional convolution to simulate “3D” lighting.

5. Cartoon Filter

Bilateral filtering + adaptive threshold.

Produces stylized cartoon-like images.

6. Threshold Filter

Converts image to grayscale and applies binary threshold.

7. HSV Filter

Demonstrates color-space conversion (RGB → HSV).

Extracts a user-defined color range.

8. Invert Filter

Simple pixel inversion.

Useful for understanding direct pixel operations.

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
