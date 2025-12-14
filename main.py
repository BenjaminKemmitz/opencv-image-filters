import argparse
import cv2
import sys
import os
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim

from filters.gaussian_blur import gaussian_blur

# ----------------------------
# Filter registry
# ----------------------------
FILTERS = {
    "blur": gaussian_blur,
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

def calculate_psnr(original, filtered):
    mse = np.mean(
        (original.astype(np.float32) - filtered.astype(np.float32)) ** 2
    )
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255 ** 2) / mse)

def calculate_ssim(original, filtered):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(original_gray, filtered_gray, full=True)
    return score

def edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return (edges > 0).sum() / edges.size

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

    args = parser.parse_args()

    if args.list:
        list_filters()
        sys.exit(0)

    filter_name = args.filter.lower()
    if filter_name not in FILTERS:
        print(f"Error: Unknown filter '{filter_name}'")
        list_filters()
        sys.exit(1)

    # Load image
    original = cv2.imread(args.image)
    if original is None:
        print(f"Error: Could not load image '{args.image}'")
        sys.exit(1)

    filter_func = FILTERS[filter_name]

    # Apply filter
    filtered = filter_func(original)

    # ----------------------------
    # Metrics
    # ----------------------------
    psnr_value = calculate_psnr(original, filtered)
    ssim_value = calculate_ssim(original, filtered)
    edge_val = edge_density(filtered)
    runtime_ms = measure_runtime(filter_func, original) * 1000

    print("\n--- Filter Evaluation Metrics ---")
    print(f"Filter: {filter_name}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"Edge Density: {edge_val:.4f}")
    print(f"Runtime: {runtime_ms:.2f} ms")

    # ----------------------------
    # Save output
    # ----------------------------
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_root, "images", "output")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{filter_name}_output.jpg")
    cv2.imwrite(output_path, filtered)
    print(f"\nSaved output to {output_path}")

    # ----------------------------
    # Display
    # ----------------------------
    if not args.no_display:
        combined = cv2.hconcat([
            resize_for_display(original),
            resize_for_display(filtered)
        ])
        cv2.imshow("Original | Filtered", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
