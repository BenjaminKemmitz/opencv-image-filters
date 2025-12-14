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

FILTERS = {
    "blur": gaussian_blur,
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
