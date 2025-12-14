import argparse
import cv2
import sys

def resize_for_display(image, max_width=800):
    h, w = image.shape[:2]
    if w <= max_width:
        return image

    scale = max_width / w
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

from filters.gaussian_blur import gaussian_blur
from filters.adaptive_threshold import adaptive_threshold
from filters.bilateral_filter import bilateral_filter
from filters.canny_edge import canny_edge
from filters.custom_kernel import custom_filter
from filters.fast_features import fast_features
from filters.harris_corners import harris_corners
from filters.hsv_color_segment import hsv_segment
from filters.laplacian_edge import laplacian_edge
from filters.median import median
from filters.morphology import morphology
from filters.orb_features import orb_features
from filters.otsu_thresholding import otsu_threshold
from filters.shi_tomasi import shi_tomasi
from filters.sobel_edge import sobel_edge
from filters.thresholding import basic_threshold

FILTERS = {
    "blur": gaussian_blur,
    "a-threshold": adaptive_threshold,
    "bilateral": bilateral_filter,
    "canny": canny_edge,
    "custom": custom_kernel,
    "fast": fast_features,
    "harris": harris_corners,
    "hsv": hsv_color_segment,,
    "laplacian": laplacian_edge,
    "median": median,
    "morphology": morphology,
    "orb": orb_features,
    "otsu": otsu_thresholding,
    "shi-tomasi": shi_tomasi,
    "sobel": sobel_edge,
    "thresholding": thresholding,
}


def list_filters():
    print("Available filters:")
    for f in FILTERS.keys():
        print(" -", f)


def main():
    parser = argparse.ArgumentParser(description="Apply image filters using OpenCV.")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--filter",
        type=str,
        required=True,
        help="Which filter to apply (use --list to see options)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available filters and exit",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        list_filters()
        sys.exit(0)

    filter_name = args.filter.lower()

    if filter_name not in FILTERS:
        print(f"Error: Unknown filter '{filter_name}'.")
        list_filters()
        sys.exit(1)

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image '{args.image}'")
        sys.exit(1)

    # Apply filter
    result = FILTERS[filter_name](img)
    import os

    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_root, "images", "output")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{filter_name}_output.jpg")

    success = cv2.imwrite(output_path, result)
    if success:
        print(f"Saved output to {output_path}")
    else:
        print(f"ERROR: Failed to save image to {output_path}")

    print("Result type:", type(result))
    print("Result shape:", result.shape if result is not None else None)
    print("Result dtype:", result.dtype if result is not None else None)

    # Resize for display
    display_img = resize_for_display(result)

    cv2.imshow("Output", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
