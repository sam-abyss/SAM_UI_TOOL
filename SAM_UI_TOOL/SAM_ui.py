#!/usr/bin/env python3
import tkinter
from tkinter import ttk, messagebox
from typing import Tuple, List
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import pathlib
import os
import argparse

from segment_anything import sam_model_registry, SamPredictor


class SAMForm:
    def __init__(self, args, master, images: List[str]) -> None:
        self.master = master
        self.images = images
        master.title("SAM Tool")
        self.buttons = tkinter.Frame(master)
        self.reset_button = ttk.Button(self.buttons, text="Reset", command=self.reset)
        self.reset_button.grid(row=0, column=0)
        self.submit_button = ttk.Button(
            self.buttons, text="Generate Mask", command=self.submit
        )
        self.submit_button.grid(row=2, column=1)
        self.previous_button = ttk.Button(
            self.buttons, text="Previous", command=self.previous
        )
        self.previous_button.grid(row=0, column=1)
        self.done_button = ttk.Button(self.buttons, text="Done", command=self.done)
        self.done_button.grid(row=0, column=2)
        self.exit_button = ttk.Button(self.buttons, text="Exit", command=master.quit)
        self.exit_button.grid(row=0, column=3)
        self.buttons.grid(row=0, column=0)
        self.image_num = 0
        self.image = Image.open(self.images[self.image_num])
        if self.image.size[0] > 2000:
            self.image = self.image.resize(
                (2000, int(self.image.size[1] * 2000 / self.image.size[0]))
            )

        self.draw = ImageDraw.Draw(self.image)
        self.base_image = Image.open(self.images[self.image_num])
        if self.base_image.size[0] > 2000:
            self.base_image = self.base_image.resize(
                (2000, int(self.base_image.size[1] * 2000 / self.base_image.size[0]))
            )
        self.draw_base = ImageDraw.Draw(self.base_image)
        self.scan = ImageTk.PhotoImage(self.image)
        self.label = tkinter.Label(master, image=self.scan)
        self.label.image = self.scan
        self.label.grid(row=1, column=0)
        self.label2 = tkinter.Label(
            master, text=f"image {self.image_num + 1} of {len(self.images)}"
        )
        self.label2.grid(row=2, column=0)
        self.label.bind("<Button-1>", self.left_click)
        self.label.bind("<Button-3>", self.right_click)
        self.points_list = []
        self.labels = []
        self.maskImg = None
        self.overwrite = False
        self.sam_checkpoint = args.model_path
        self.out_dir = args.output_dir
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device="cuda" if args.cuda else "cpu")
        self.predictor = SamPredictor(self.sam)

    def position(self, event: tkinter.Event) -> Tuple[int, int]:
        x = event.x
        y = event.y
        return (x, y)

    def submit(self):
        mask, scores, logits = self.generate_masks()
        mask, _, _ = self.generate_masks(scores=scores, logits=logits)
        maskArr = np.array(mask[0], dtype=np.uint8) * 255
        self.maskImg = Image.fromarray(maskArr, mode="L")

        image2 = self.maskImg

        self.draw_mask(image2)

    def left_click(self, event: tkinter.Event) -> None:
        x, y = self.position(event)
        print(f"left click at {x},{y}")
        self.points_list.append((x, y))
        self.labels.append(1)
        self.draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill="green")
        self.draw_base.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill="green")
        self.new_image(self.image)

        print(self.points_list, self.labels)

    def right_click(self, event: tkinter.Event) -> None:
        x, y = self.position(event)
        print(f"right click at {x},{y}")
        self.points_list.append((x, y))
        self.labels.append(0)
        self.draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill="red")
        self.draw_base.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill="red")
        self.new_image(self.image)

        print(self.points_list, self.labels)

    def generate_masks(self, logits=None, scores=None):
        input_points = np.asarray(self.points_list)
        input_labels = np.asarray(self.labels)

        self.predictor.set_image(cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR))

        try:
            mask_input = logits[np.argmax(scores), :, :]
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )

        except:
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )

        return masks, scores, logits

    def draw_mask(self, mask):
        if self.base_image.size != mask.size or self.base_image.mode != mask.mode:
            mask = mask.resize(self.base_image.size)
            mask = mask.convert(self.base_image.mode)

        im3 = Image.blend(self.base_image, mask, 0.3)

        self.new_image(im3)

    def new_image(self, image) -> None:
        self.label.grid_forget()
        self.image = image
        self.draw = ImageDraw.Draw(self.image)
        if self.image.size[0] > 2000:
            self.image = self.image.resize(
                (2000, int(self.image.size[1] * 2000 / self.image.size[0]))
            )
        self.scan = ImageTk.PhotoImage(self.image)
        self.label = tkinter.Label(self.master, image=self.scan)
        self.label.image = self.scan
        self.label.grid(row=1, column=0)
        self.label.bind("<Button-1>", self.left_click)
        self.label.bind("<Button-3>", self.right_click)
        self.label2.grid_forget()
        self.label2 = tkinter.Label(
            self.master, text=f"image {self.image_num + 1} of {len(self.images)}"
        )
        self.label2.grid(row=2, column=0)

    def done(self) -> None:
        ext = pathlib.Path(self.images[self.image_num]).suffix
        file_name = (
            os.path.basename(self.images[self.image_num]).strip(ext) + "_mask.png"
        )
        out_path = os.path.join(self.out_dir, file_name)

        if self.overwrite or not os.path.exists(out_path):
            self.maskImg.save(out_path)
        else:
            messagebox.showwarning(
                message="Mask file exists. Set overwrite flag if you want to update new mask."
            )

        self.points_list = []
        self.labels = []

        self.image_num += 1
        if self.image_num < len(self.images):
            self.new_image(Image.open(self.images[self.image_num]))
            self.base_image = Image.open(self.images[self.image_num])
            if self.base_image.size[0] > 2000:
                self.base_image = self.base_image.resize(
                    (
                        2000,
                        int(self.base_image.size[1] * 2000 / self.base_image.size[0]),
                    )
                )
            self.draw_base = ImageDraw.Draw(self.base_image)
        else:
            self.master.quit()

    def reset(self) -> None:
        self.points_list = []
        self.labels = []
        self.new_image(Image.open(self.images[self.image_num]))
        self.base_image = Image.open(self.images[self.image_num])
        self.draw_base = ImageDraw.Draw(self.base_image)

    def previous(self):
        self.points_list = []
        self.labels = []
        if self.image_num > 0:
            self.image_num -= 1
            self.new_image(Image.open(self.images[self.image_num]))
            self.base_image = Image.open(self.images[self.image_num])
            if self.base_image.size[0] > 2000:
                self.base_image = self.base_image.resize(
                    (
                        2000,
                        int(self.base_image.size[1] * 2000 / self.base_image.size[0]),
                    )
                )
            self.draw_base = ImageDraw.Draw(self.base_image)
        else:
            messagebox.showwarning(message="You are annotating the first image")


def get_args():
    """
    Defines and parses command-line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-images",
        "--input-images-path-file",
        "--input",
        type=str,
        help="Text file containing list of image paths",
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
    with open(args.input_images, "r") as file:
        images = [line.rstrip() for line in file]
    SAMForm(args=args, master=root, images=images)
    root.mainloop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
