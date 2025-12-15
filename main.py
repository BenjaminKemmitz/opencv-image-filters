import argparse
import cv2
import sys
import os
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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
# Noise
# ---------------------------
def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, amount=0.02):
    noisy = image.copy()
    num_pixels = int(amount * image.shape[0] * image.shape[1])

    for _ in range(num_pixels):
        y = np.random.randint(0, image.shape[0])
        x = np.random.randint(0, image.shape[1])
        noisy[y, x] = 255 if np.random.rand() < 0.5 else 0

    return noisy

def noise_robustness_benchmark(image, filters, noise_type="gaussian"):
    sigmas = [5, 10, 20, 30, 40, 50]
    rows = []

    for sigma in sigmas:
        if noise_type == "gaussian":
            noisy = add_gaussian_noise(image, sigma)
        else:
            noisy = add_salt_pepper_noise(image, amount=sigma / 500)

        for name, func in filters.items():
            try:
                filtered = func(noisy)

                if filtered.ndim == 2:
                    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

                rows.append({
                    "filter": name,
                    "noise_level": sigma,
                    "psnr": compute_psnr(image, filtered),
                    "ssim": compute_ssim(image, filtered)
                })
            except Exception:
                continue

    return pd.DataFrame(rows)

def plot_noise_robustness(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for metric in ["psnr", "ssim"]:
        plt.figure(figsize=(8, 5))

        for f in df["filter"].unique():
            subset = df[df["filter"] == f]
            plt.plot(subset["noise_level"], subset[metric], label=f)

        plt.xlabel("Noise Level (σ)")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} vs Noise Level")
        plt.legend(fontsize=7)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"noise_{metric}.png"))
        plt.close()
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.count_nonzero(edges) / edges.size

# ----------------------------
# Display helper
# ----------------------------
def resize_for_display(image, max_width=800):
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

# ----------------------------
# Plotting
# ----------------------------
def generate_plots(csv_path):
    df = pd.read_csv(csv_path)
    plot_dir = os.path.join(os.path.dirname(csv_path), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    metrics = {
        "psnr": "PSNR",
        "ssim": "SSIM",
        "edge_density": "Edge Density",
        "runtime_ms": "Runtime (ms)"
    }

    for key, label in metrics.items():
        df_sorted = df.sort_values(by=key, ascending=False)
        plt.figure(figsize=(10, 5))
        plt.bar(df_sorted["filter"], df_sorted[key])
        plt.ylabel(label)
        plt.xlabel("Filter")
        plt.title(f"{label} by Filter")
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = os.path.join(plot_dir, f"{key}.png")
        plt.savefig(path)
        plt.close()

# ----------------------------
# Markdown report
# ----------------------------
def generate_markdown_report(csv_path, image_name):
    df = pd.read_csv(csv_path)
    report_dir = os.path.dirname(csv_path)
    report_name = os.path.basename(csv_path).replace("metrics_", "REPORT_").replace(".csv", ".md")
    report_path = os.path.join(report_dir, report_name)

    best_psnr = df.loc[df["psnr"].idxmax()]
    best_ssim = df.loc[df["ssim"].idxmax()]
    fastest = df.loc[df["runtime_ms"].idxmin()]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# OpenCV Filter Experiment Report\n\n")
        f.write(f"**Input Image:** `{image_name}`\n\n")
        f.write("## Summary\n")
        f.write(f"- Best PSNR: **{best_psnr['filter']}** ({best_psnr['psnr']:.2f})\n")
        f.write(f"- Best SSIM: **{best_ssim['filter']}** ({best_ssim['ssim']:.4f})\n")
        f.write(f"- Fastest Filter: **{fastest['filter']}** ({fastest['runtime_ms']:.2f} ms)\n\n")
        f.write("## Metrics Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

# ----------------------------
# Dataset processing
# ----------------------------
def process_dataset(dataset_dir, filters):
    rows = []

    for image_name in os.listdir(dataset_dir):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = cv2.imread(os.path.join(dataset_dir, image_name))
        if img is None:
            continue

        for name, func in filters.items():
            try:
                start = time.perf_counter()
                result = func(img)
                runtime_ms = (time.perf_counter() - start) * 1000

                if not isinstance(result, np.ndarray):
                    continue

                if result.ndim == 2:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

                rows.append({
                    "image": image_name,
                    "filter": name,
                    "psnr": compute_psnr(img, result),
                    "ssim": compute_ssim(img, result),
                    "edge_density": compute_edge_density(result),
                    "runtime_ms": runtime_ms
                })
            except Exception as e:
                print(f"Filter failed: {name} on {image_name} — {e}")

    return pd.DataFrame(rows)

def aggregate_dataset_metrics(df):
    return (
        df.groupby("filter")
        .agg(
            psnr_mean=("psnr", "mean"),
            ssim_mean=("ssim", "mean"),
            edge_density_mean=("edge_density", "mean"),
            runtime_mean=("runtime_ms", "mean"),
            psnr_std=("psnr", "std"),
            ssim_std=("ssim", "std")
        )
        .reset_index()
    )

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--filter")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--noise-benchmark", action="store_true")
    args = parser.parse_args()

    if args.list:
        for f in FILTERS:
            print(f)
        sys.exit(0)

    if not args.dataset and not args.all and not args.filter:
        print("Error: specify --filter, --all, or --dataset")
        sys.exit(1)

    project_root = os.path.dirname(os.path.abspath(__file__))

    if args.noise_benchmark:
        img = cv2.imread(args.image)
        if img is None:
            print("Error loading image")
            sys.exit(1)
    
        out_dir = os.path.join(project_root, "experiments", "noise")
        os.makedirs(out_dir, exist_ok=True)
    
        df = noise_robustness_benchmark(img, FILTERS, noise_type="gaussian")
        csv_path = os.path.join(out_dir, "noise_metrics.csv")
        df.to_csv(csv_path, index=False)
    
        plot_noise_robustness(df, out_dir)
        print("Noise robustness benchmark complete.")
        sys.exit(0)
    
    # -------- DATASET MODE --------
    if args.dataset:
        df = process_dataset(args.dataset, FILTERS)
        out_dir = os.path.join(project_root, "experiments", "dataset")
        os.makedirs(out_dir, exist_ok=True)

        df.to_csv(os.path.join(out_dir, "dataset_metrics.csv"), index=False)
        agg = aggregate_dataset_metrics(df)
        agg.to_csv(os.path.join(out_dir, "aggregated_metrics.csv"), index=False)
        sys.exit(0)

    # -------- SINGLE IMAGE MODE --------
    img = cv2.imread(args.image)
    if img is None:
        print("Error loading image")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(project_root, "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    metrics_path = os.path.join(exp_dir, f"metrics_{timestamp}.csv")

    filters_to_run = FILTERS.keys() if args.all else [args.filter]

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "filter", "psnr", "ssim", "edge_density", "runtime_ms"])

        for name in filters_to_run:
            try:
                result = FILTERS[name](img)
                if result.ndim == 2:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

                writer.writerow([
                    os.path.basename(args.image),
                    name,
                    compute_psnr(img, result),
                    compute_ssim(img, result),
                    compute_edge_density(result),
                    0
                ])
            except Exception as e:
                print(f"Filter failed: {name} — {e}")

    generate_plots(metrics_path)
    generate_markdown_report(metrics_path, os.path.basename(args.image))

if __name__ == "__main__":
    main()
