import argparse
import cv2
import sys
import os
import time
import csv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datetime import datetime

# ----------------------------
# Import filters
# ----------------------------
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
    "adaptive-threshold": adaptive_threshold,
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
    "threshold": basic_threshold,
}

# ----------------------------
# Metrics
# ----------------------------
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
    return np.count_nonzero(edges) / edges.size

# ----------------------------
# Display helper
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

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="OpenCV filter experiment runner with metrics"
    )
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--filter", help="Single filter name")
    parser.add_argument("--all", action="store_true", help="Run all filters")
    parser.add_argument("--list", action="store_true", help="List filters and exit")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI")

    args = parser.parse_args()

    if args.list:
        print("Available filters:")
        for f in FILTERS:
            print(" -", f)
        sys.exit(0)

    if not args.all and not args.filter:
        print("Error: specify --filter or --all")
        sys.exit(1)

    # Paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_root, "images", "output")
    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(project_root, "experiments")
    os.makedirs(metrics_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metrics_path = os.path.join(metrics_dir, f"metrics_{timestamp}.csv")

    write_header = True  # new file every run
 
    # Load image
    original = cv2.imread(args.image)
    if original is None:
        print(f"Error: Could not load image '{args.image}'")
        sys.exit(1)

    filters_to_run = FILTERS.keys() if args.all else [args.filter.lower()]

    # --- SAFE METRICS CSV OPEN ---
    try:
        csvfile = open(metrics_path, "a", newline="")
    except PermissionError:
        print("\nERROR: metrics.csv is locked.")
        print("Close Excel / any editor using the file and re-run.")
        sys.exit(1)

    with csvfile:
        writer = csv.writer(csvfile)

        if write_header:
            writer.writerow([
                "image",
                "filter",
                "psnr",
                "ssim",
                "edge_density",
                "runtime_ms"
            ])

        for name in filters_to_run:
            if name not in FILTERS:
                print(f"Skipping unknown filter: {name}")
                continue

            print(f"\nApplying filter: {name}")
            filter_func = FILTERS[name]

            start = time.perf_counter()
            result = filter_func(original)
            runtime_ms = (time.perf_counter() - start) * 1000

            output_path = os.path.join(output_dir, f"{name}_output.jpg")
            cv2.imwrite(output_path, result)

            psnr_val = compute_psnr(original, result)
            ssim_val = compute_ssim(original, result)
            edge_val = compute_edge_density(result)

            writer.writerow([
                os.path.basename(args.image),
                name,
                f"{psnr_val:.4f}",
                f"{ssim_val:.4f}",
                f"{edge_val:.6f}",
                f"{runtime_ms:.2f}"
            ])
    
            print(
                f"PSNR={psnr_val:.2f} | "
                f"SSIM={ssim_val:.4f} | "
                f"Edges={edge_val:.5f} | "
                f"Time={runtime_ms:.2f}ms"
            )
    
            if not args.no_display and not args.all:
                combined = cv2.hconcat([
                    resize_for_display(original),
                    resize_for_display(result)
                ])
                cv2.imshow("Original | Filtered", combined)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
