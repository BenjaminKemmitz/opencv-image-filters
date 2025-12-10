import argparse
import cv2
import sys

from filters.blur_filter import apply_blur
from filters.sharpen_filter import apply_sharpen
from filters.edge_detect_filter import apply_edge_detect
from filters.emboss_filter import apply_emboss
from filters.cartoon_filter import apply_cartoon
from filters.threshold_filter import apply_threshold
from filters.hsv_filter import apply_hsv
from filters.invert_filter import apply_invert


FILTERS = {
    "blur": apply_blur,
    "sharpen": apply_sharpen,
    "edge": apply_edge_detect,
    "emboss": apply_emboss,
    "cartoon": apply_cartoon,
    "threshold": apply_threshold,
    "hsv": apply_hsv,
    "invert": apply_invert,
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

    # Display
    cv2.imshow("Input", img)
    cv2.imshow("Output", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
