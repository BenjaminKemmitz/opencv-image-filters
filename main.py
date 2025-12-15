import argparse
import cv2
import sys
import os
import time
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from filters.gaussian_blur import gaussian_blur
from filters.adaptive_threshold import adaptive_threshold
from filters.bilateral_filter import bilateral_filter
from filters.canny_edge import canny_edge
from filters.custom_kernel import custom_kernel
from filters.fast_features import fast_features
from filters.harris_corners import harris_corners
from filters.hsv_color_segment import hsv_segment
from filters.laplacian_edge import laplacian_edge
from filters.median import median
from filters.morphology import morphology
from filters.orb_features import orb_features
from filters.otsu_thresholding import otsu_thresholding
from filters.shi_tomasi import shi_tomasi
from filters.sobel_edge import sobel_edge
from filters.thresholding import basic_threshold


# ----------------------------
# Filter registry
# ----------------------------
FILTERS = {
    "blur": gaussian_blur,
    "a-threshold": adaptive_threshold,
    "bilateral": bilateral_filter,
    "canny": canny_edge,
    "custom": custom_kernel,
    "fast": fast_features,
    "harris": harris_corners,
    "hsv": hsv_segment,
    "laplacian": laplacian_edge,
    "median": median,
    "morphology": morphology,
    "orb": orb_features,
    "otsu": otsu_thresholding,
    "shi-tomasi": shi_tomasi,
    "sobel": sobel_edge,
    "thresholding": basic_threshold,
}

# ----------------------------
# Utility functions
# ----------------------------
def resize_for_display(image, max_width=800):
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )

def compute_psnr(original, filtered):
    return peak_signal_noise_ratio(original, filtered, data_range=255)

def compute_ssim(original, filtered):
     return structural_similarity(
        cv2.cvtColor(original, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY),
        data_range=255
    )
    
def compute_edge_density(image):
    edges = cv2.Canny(image, 100, 200)
    return np.sum(edges > 0) / edges.size

def measure_runtime(filter_func, image, runs=10):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        filter_func(image)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)

def list_filters():
    print("Available filters:")
    for f in FILTERS:
        print(" -", f)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Apply OpenCV filters with quantitative evaluation"
    )
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--filter", type=str, required=True, help="Filter name")
    parser.add_argument("--list", action="store_true", help="List filters and exit")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI output")
    parser.add_argument( "--all", action="store_true", help="Apply all available filters to the input image")
    args = parser.parse_args()

    if args.list:
        list_filters()
        sys.exit(0)

    if not args.all:
    filter_name = args.filter.lower()

    if filter_name not in FILTERS:
        print(f"Error: Unknown filter '{filter_name}'.")
        list_filters()
        sys.exit(1)

    import os
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Load image
    original = cv2.imread(args.image)
    if original is None:
        print(f"Error: Could not load image '{args.image}'")
        sys.exit(1)

   

    # Apply filter
    filtered = filter_func(original)

    # ----------------------------
    # Metrics
    # ----------------------------
    psnr = compute_psnr(original, filtered)
    ssim = compute_ssim(original, filtered)
    edge_density = compute_edge_density(filtered)
    runtime = measure_runtime(filter_func, original) * 1000

    print("\n--- Filter Evaluation Metrics ---")
    print(f"Filter: {filter_name}")
    print(f"PSNR: {psnr:.2f}")
    print(f"SSIM: {ssim:.4f}")
    print(f"Edge Density: {edge_density:.4f}")
    print(f"Runtime: {runtime:.6f} seconds")

    metrics_path = os.path.join(project_root, "experiments", "metrics.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    row = {
        "image": os.path.basename(args.image),
        "filter": filter_name,
        "psnr": psnr,
        "ssim": ssim,
        "edge_density": edge_density,
        "runtime_sec": runtime
    }

    df = pd.DataFrame([row])

    if os.path.exists(metrics_path):
        df.to_csv(metrics_path, mode="a", header=False, index=False)
    else:
        df.to_csv(metrics_path, index=False)
    # ----------------------------
    # Save output
    # ----------------------------
    output_dir = os.path.join(project_root, "images", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    import time

    filters_to_run = FILTERS.keys() if args.all else [filter_name]

    for name in filters_to_run:
        print(f"Applying filter: {name}")

        start = time.perf_counter()
        result = FILTERS[name](img)
        elapsed = time.perf_counter() - start

        output_path = os.path.join(output_dir, f"{name}_output.jpg")
        success = cv2.imwrite(output_path, result)

        if success:
            print(f"  Saved: {output_path} ({elapsed:.4f}s)")
        else:
            print(f"  ERROR saving {name}")

    # ----------------------------
    # Display
    # ----------------------------
    if not args.no_display:
        combined = cv2.hconcat([
            resize_for_display(original),
            resize_for_display(filtered)
        ])
        if not args.all:
            display_img = resize_for_display(result)
            cv2.imshow("Original", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
