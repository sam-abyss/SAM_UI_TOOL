#!/usr/bin/env python3
import argparse
import tkinter
import os
import cv2
from SAM_image_segmentation import SAMForm

DESCRIPTION = """
Application for generating segmentation masks using Segment Anything Model (SAM).
Uses Tkinter, which is a standard GUI (Graphical User Interface) library for Python.

Usage: SAM-ui [Options]

Options:
  --input-folder TEXT     Path to folder containing images to be masked.
  --images-file TEXT      Text file containing list of absolute image paths.
  --model-path TEXT       Path to downloaded Segment Anything model. Refer to
                          https://github.com/facebookresearch/segment-anything#model-checkpoints
                          to download ViT-H SAM model.
  --output-dir TEXT       Path to output directory to save masks in .png format.
  --cuda                  Use this flag to process using CUDA (if you have CUDA setup).

Examples:
  SAM-ui --input-folder ./input_images/ --model-path ./models/vit_h_384.pth --output-dir ./output_masks/
  SAM-ui --images-file ./image_list.txt --model-path ./models/vit_h_384.pth --output-dir ./output_masks/ --cuda
"""


def get_args() -> argparse.Namespace:
    """
    Defines and parses command-line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        help="Path to folder containing images to be masked",
    )
    parser.add_argument(
        "--images-file",
        type=str,
        help="Text file containing list of absolute image paths",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to downloaded Segment Anything model. Refer https://github.com/facebookresearch/segment-anything#model-checkpoints. Download ViT-H SAM model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory to save masks in .png format",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use this flag to process using CUDA (if you have CUDA setup)",
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()
    root = tkinter.Tk()
    if args.input_folder:
        images = [
            os.path.abspath(os.path.join(args.input_folder, p))
            for p in os.listdir(args.input_folder)
        ]
    elif args.images_file:
        with open(args.images_file, "r") as file:
            images = [line.rstrip() for line in file]
    SAMForm(args=args, master=root, images=images)
    root.mainloop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
