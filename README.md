# OpenCV Image Filters Benchmark Suite

A research-oriented computer vision project that implements, benchmarks, and analyzes classical OpenCV image-processing filters under clean, noisy, and dataset-scale conditions.

This project goes beyond simple demos by providing:

* Quantitative metrics (PSNR, SSIM, edge density, runtime)
* Dataset-level aggregation
* Noise robustness benchmarking
* Automated plots and Markdown experiment reports

Designed as a portfolio-quality and research-preparatory project for robotics and computer vision work.

---

## Features

* 16 classical OpenCV filters implemented in a modular architecture
* Command-line interface (CLI) with multiple execution modes
* Automatic experiment logging to CSV
* Noise robustness benchmarking (Gaussian noise)
* Dataset-level evaluation with mean / std aggregation
* Publication-style plots (PSNR, SSIM, runtime, edge density)
* Auto-generated Markdown experiment reports

---

## Project Structure

```
opencv-image-filters/
│── filters/                # Individual filter implementations
│── images/
│   ├── input/              # Input images
│   └── output/             # Filtered outputs
│── experiments/
│   ├── noise/              # Noise robustness experiments
│   ├── dataset/            # Dataset-level experiments
│   └── metrics_*.csv       # Per-run metrics
│── main.py                 # Main CLI entry point
│── requirements.txt
│── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BenjaminKemmitz/opencv-image-filters.git
cd opencv-image-filters
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage (CLI)

### List available filters

```bash
python main.py --list
```

### Run a single filter

```bash
python main.py --image images/input/sunrise.jpeg --filter blur
```

### Run all filters on one image

```bash
python main.py --image images/input/sunrise.jpeg --all --no-display
```

### Run dataset-level benchmarking

```bash
python main.py --dataset images/input
```

### Run noise robustness benchmarking

```bash
python main.py --image images/input/sunrise.jpeg --noise-benchmark
```

---

## Implemented Filters

* Gaussian Blur
* Median Filter
* Bilateral Filter
* Sobel Edge Detector
* Laplacian Edge Detector
* Canny Edge Detector
* Morphological Operations
* HSV Color Segmentation
* Basic Thresholding
* Adaptive Thresholding
* Otsu’s Thresholding
* Harris Corner Detector
* Shi–Tomasi Feature Detector
* FAST Corner Detector
* ORB Feature Detector
* Custom Sharpening Filter

---

## Metrics Explained

| Metric           | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| **PSNR**         | Measures signal fidelity between original and filtered images |
| **SSIM**         | Structural similarity index (perceptual quality)              |
| **Edge Density** | Fraction of edge pixels (Canny-based)                         |
| **Runtime (ms)** | Filter execution time                                         |

---

## Noise Robustness Benchmarking

The noise benchmark evaluates filter performance under increasing Gaussian noise levels.

For each noise level (σ = 5 → 50):

1. Noise is added to the original image
2. Each filter is applied
3. PSNR and SSIM are measured against the clean image

### Outputs

```
experiments/noise/
├── noise_metrics.csv
├── noise_psnr.png
├── noise_ssim.png
```

These plots visualize robustness vs noise strength, a common evaluation in computer vision research.

---

## Dataset-Level Experiments

When run in dataset mode, the system:

* Processes all images in a directory
* Logs per-image metrics
* Aggregates mean and standard deviation per filter

Outputs:

```
experiments/dataset/
├── dataset_metrics.csv
├── aggregated_metrics.csv
```

---

## Automated Reports

Each experiment generates a Markdown report containing:

* Summary of best-performing filters
* Metrics table
* Key quantitative findings

This enables reproducible, documented experimentation.

---

## Key Findings (Example)

* Gaussian blur achieves the highest PSNR under Gaussian noise
* Bilateral filtering preserves edges better at the cost of runtime
* Edge detectors degrade rapidly as noise increases
* Classical filters show clear trade-offs between speed and fidelity

---

## Project Goals

This project was built to:

* Demonstrate strong foundations in classical computer vision
* Practice experimental design and benchmarking
* Prepare for robotics and vision research
* Serve as a portfolio-ready, grad-school-quality project

---

## Future Work

This project is considered complete. Possible future directions are intentionally left to separate projects, such as:

* Learning-based denoising (CNNs)
* Vision-guided robotics systems
* Real-time performance optimization

---

## Author

**Ben Kemmitz**
Robotics • Computer Vision • Python

* GitHub: [https://github.com/BenjaminKemmitz](https://github.com/BenjaminKemmitz)
* Website: [benjaminkemmitz.com](benjaminkemmitz.com)
* LinkedIn: [https://www.linkedin.com/in/ben-kemmitz-a50442399/](https://www.linkedin.com/in/ben-kemmitz-a50442399/)

---

## License

This project is released for educational and portfolio use.
