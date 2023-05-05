#!/usr/bin/env python3
"""
This module provides a Python tool for generating masks used in photogrammetry.
The graphical user interface (GUI) is built with Tkinter and utilizes the Segment Anything Meta (SAM) model to generate masks.

Usage:
   - Launch the GUI by running the script.
   - Click on the image to mark the regions that should be includes or excluded (left and right clicks)
   - Click on the 'Generate Mask' button to generate the mask for the marked regions.
   - The generated mask can be saved by clicking on the 'Done' button.
"""
import tkinter
from tkinter import ttk, messagebox
from typing import Tuple, List, Any
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import pathlib
import os

from segment_anything import sam_model_registry, SamPredictor


class SAMForm:
    def __init__(self, args, master, images: List[str]) -> None:
        """
        The SAMForm class is a tkinter-based user interface for generating masks for images using SAM model.

        Args:
            args: an object containing arguments for the SAM model
            master: the tkinter master object for the GUI
            images: a list of image paths

        Attributes:
            self.master: A tkinter master object that represents the main window of the GUI.
            self.images: A list of file paths to images that the user wants to generate masks for.
            self.buttons: A tkinter frame that contains several buttons for interacting with the GUI.
            self.reset_button: A tkinter ttk.Button object that resets the GUI to its default state.
            self.submit_button: A tkinter ttk.Button object that generates a mask for the current image.
            self.previous_button: A tkinter ttk.Button object that loads the previous image in the images list.
            self.next_button: A tkinter ttk.Button object that loads the next image in the images list.
            self.done_button: A tkinter ttk.Button object that signals that the user has finished generating masks for the current image.
            self.exit_button: A tkinter ttk.Button object that exits the GUI.
            self.image_num: An integer that represents the index of the current image in the images list.
            self.image: A PIL Image object that represents the current image.
            self.orig_resolution: A tuple that contains the original resolution of the current image.
            self.draw: A PIL ImageDraw object that is used to draw on the image object.
            self.base_image: A PIL Image object that represents the original unmodified current image.
            self.draw_base: A PIL ImageDraw object that is used to draw on the base_image object.
            self.scan: A tkinter PhotoImage object that represents the current image and is displayed in the GUI.
            self.label: A tkinter Label object that displays the scan object in the GUI.
            self.label2: A tkinter Label object that displays the index of the current image in the images list.
            self.label3: A tkinter Label object that displays the file name of the current image.
            self.points_list: A list of tuples that contains the coordinates of points that the user has clicked on the image object.
            self.labels: A list of integers that represents the labels assigned to each point in points_list.
            self.maskImg: A PIL Image object that represents the mask generated for the current image.
            self.sam_checkpoint: A string that represents the file path to the checkpoint of the segmentation model.
            self.out_dir: A string that represents the file path to the directory where the generated masks will be saved.
            self.model_type: A string that represents the type of the segmentation model.
            self.sam: A segmentation model object that is used to generate masks for images.
            self.predictor: A SamPredictor object that is used to predict masks for images.

        Methods:
            __init__(self, args, master, images: List[str]) -> None: Initializes the SAMForm object with the given arguments.

            left_click(self, event):
                Event handler for left-clicks on the scan image. Adds a point to the points_list and draws a circle on the image.

            right_click(self, event):
                Event handler for right-clicks on the scan image. Removes the most recent point from the points_list and redraws the image.

            reset(self):
                Resets the GUI to its default state.

            submit(self):
                Generates a mask for the current image using the SAM model and displays it on the base_image.

            previous(self):
                Moves to the previous image in the images list.

            next(self):
                Moves to the next image in the images list.

            done(self):
                Marks the current image as completed and saves the generated mask image to the out_dir.

            exit(self):
                Exits the GUI and closes the application.
        """
        # Tkinter UI setup
        self.master = master
        self.images = images
        master.title("SAM Tool")
        self.buttons = tkinter.Frame(master)
        self.buttons.grid(row=0, column=0)
        # Reset button
        self.reset_button = ttk.Button(self.buttons, text="Reset", command=self.reset)
        self.reset_button.grid(row=0, column=0)
        # Previous button
        self.previous_button = ttk.Button(
            self.buttons, text="Previous", command=self.previous
        )
        self.previous_button.grid(row=0, column=1)
        # Done button
        self.done_button = ttk.Button(
            self.buttons, text="Done", command=self.done, state="disabled"
        )
        self.done_button.grid(row=0, column=2)
        # Next button
        self.next_button = ttk.Button(self.buttons, text="Next", command=self.next)
        self.next_button.grid(row=0, column=3)
        # Exit button
        self.exit_button = ttk.Button(self.buttons, text="Exit", command=self.exit)
        self.exit_button.grid(row=0, column=4)
        # Generate Mask button
        self.submit_button = ttk.Button(
            self.buttons, text="Generate\nMask", command=self.submit
        )
        self.submit_button.grid(row=1, column=2)

        self.image_num = 0
        self.image = Image.open(self.images[self.image_num])
        self.orig_resolution = self.image.size
        if self.image.size[0] > 1800:
            self.image = self.image.resize(
                (1800, int(self.image.size[1] * 1800 / self.image.size[0]))
            )
        # UI image setup
        self.draw = ImageDraw.Draw(self.image)
        self.base_image = Image.open(self.images[self.image_num])
        if self.base_image.size[0] > 1800:
            self.base_image = self.base_image.resize(
                (1800, int(self.base_image.size[1] * 1800 / self.base_image.size[0]))
            )
        self.draw_base = ImageDraw.Draw(self.base_image)
        self.scan = ImageTk.PhotoImage(self.image)
        # UI labels
        self.label = tkinter.Label(master, image=self.scan)
        self.label.image = self.scan
        self.label.grid(row=1, column=0)
        self.label2 = tkinter.Label(
            master, text=f"image {self.image_num + 1} of {len(self.images)}"
        )
        self.label2.grid(row=2, column=0)
        self.label3 = tkinter.Label(
            master, text=f"{os.path.basename(self.images[self.image_num])}"
        )
        self.label3.grid(row=3, column=0)
        self.label.bind("<Button-1>", self.left_click)
        self.label.bind("<Button-3>", self.right_click)
        # SAM model setup
        self.points_list = []
        self.labels = []
        self.maskImg = None
        self.sam_checkpoint = args.model_path
        self.out_dir = args.output_dir
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device="cuda" if args.cuda else "cpu")
        self.predictor = SamPredictor(self.sam)
        self.predictor.set_image(np.array(self.image))

    def position(self, event: tkinter.Event) -> Tuple[int, int]:
        """Get the x and y coordinates of a pixel clicked by the user.

        Args:
            event: A tkinter click event object.

        Returns:
            A tuple containing the x and y coordinates of the click event.
        """
        x = event.x
        y = event.y
        return x, y

    def submit(self) -> None:
        """
        Generate a mask and display it on the canvas.
        Uses the `generate_masks` method to generate a mask based on the current
        image and segmentation model, and displays the mask on the canvas using
        the `draw_mask` method. Also enables the "Done" button.

        Returns:
            None.
        """
        _, scores, logits = self.generate_masks()
        mask, _, _ = self.generate_masks(scores=scores, logits=logits)
        maskArr = np.array(mask[0], dtype=np.uint8) * 255
        self.maskImg = Image.fromarray(maskArr, mode="L")
        image2 = self.maskImg
        self.draw_mask(image2)
        self.done_button.config(state="normal")

    def left_click(self, event: tkinter.Event) -> None:
        """
        Handle left-click events on the canvas.

        Adds a point to the list of labeled points and updates the image display to
        show the new point. Also draws a green circle at the location of the click.

        Args:
            event: A tkinter event object containing the coordinates of the click.

        Returns:
            None.
        """
        x, y = self.position(event)
        # print(f"left click at {x},{y}")
        self.points_list.append((x, y))
        self.labels.append(1)
        self.draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill="green")
        self.draw_base.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill="green")
        self.new_image(self.image)
        # print(self.points_list, self.labels)

    def right_click(self, event: tkinter.Event) -> None:
        """
        Handle right-click events on the canvas.

        Adds a point to the list of labeled points and updates the image display to
        show the new point. Also draws a red circle at the location of the click.

        Args:
            event: A tkinter event object containing the coordinates of the click.

        Returns:
            None.
        """
        x, y = self.position(event)
        # print(f"right click at {x},{y}")
        self.points_list.append((x, y))
        self.labels.append(0)
        self.draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill="red")
        self.draw_base.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill="red")
        self.new_image(self.image)
        # print(self.points_list, self.labels)

    def generate_masks(self, logits=None, scores=None) -> Any:
        """
        Generate binary masks for the input image.

        Uses segment anything model by Meta (SAM) to predict a binary mask for each labeled
        point on the input image. If `logits` and `scores` are provided, they are used to
        guide the mask prediction process.

        Args:
            logits: A tensor containing the logits (unnormalized probabilities) output
                by the model.
            scores: A tensor containing the scores (normalized probabilities) output
                by the model.

        Returns:
            A tuple containing the binary masks generated by the model, as well as the
            scores and logits computed during the prediction process.
        """
        input_points = np.asarray(self.points_list)
        input_labels = np.asarray(self.labels)
        try:
            mask_input = logits[np.argmax(scores), :, :]
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )
        except TypeError:
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )

        return masks, scores, logits

    def draw_mask(self, mask) -> None:
        """
        Draws a binary mask over the base image.

        Resizes the input binary mask if necessary, then blends it with the base image
        using a fixed opacity of 0.3. The resulting image is displayed in the GUI.

        Args:
            mask: A binary mask image to draw over the base image.

        Returns:
            None
        """
        if self.base_image.size != mask.size or self.base_image.mode != mask.mode:
            mask = mask.resize(self.base_image.size)
            mask = mask.convert(self.base_image.mode)

        im3 = Image.blend(self.base_image, mask, 0.3)

        self.new_image(im3)

    def new_image(self, image) -> None:
        """
        Updates the GUI with a new image.

        Removes the current image label from the GUI, sets the input image as the new
        image to display, resizes it if necessary, and creates a new label to display
        the image. Also updates the image number and filename labels, and disables the
        "Done" button.

        Args:
            image: The new image to display in the GUI.

        Returns:
            None
        """
        self.label.grid_forget()
        self.image = image
        self.draw = ImageDraw.Draw(self.image)
        if self.image.size[0] > 1800:
            self.image = self.image.resize(
                (1800, int(self.image.size[1] * 1800 / self.image.size[0]))
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
        self.label3.grid_forget()
        self.label3 = tkinter.Label(
            self.master, text=f"{os.path.basename(self.images[self.image_num])}"
        )
        self.label3.grid(row=3, column=0)
        self.done_button.config(state="disable")

    def done(self) -> None:
        """
        The done method saves the masked image in the output directory and resets the
        points list and labels. If the output file exists, it prompts the user to
        confirm if they want to overwrite the existing file. After that, it loads the
        next image to be processed and sets it as the base image. If there are no more
        images left to process, it quits the application.

        Returns:
            None
        """

        ext = pathlib.Path(self.images[self.image_num]).suffix
        file_name = (
            os.path.basename(self.images[self.image_num]).strip(ext) + "_mask.png"
        )
        out_path = os.path.join(self.out_dir, file_name)

        if not os.path.exists(out_path):
            resized_image = self.maskImg.resize(self.orig_resolution)
            resized_image.save(out_path)
        else:
            response = messagebox.askquestion(
                "Confirmation", "Mask file exists. Do you want to overwrite?"
            )
            if response == "yes":
                resized_image = self.maskImg.resize(self.orig_resolution)
                resized_image.save(out_path)
            elif response == "no":
                pass

        self.points_list = []
        self.labels = []

        self.image_num += 1
        if self.image_num < len(self.images):
            self.new_image(Image.open(self.images[self.image_num]))
            self.base_image = Image.open(self.images[self.image_num])
            self.predictor.set_image(np.array(self.image))
            if self.base_image.size[0] > 1800:
                self.base_image = self.base_image.resize(
                    (
                        1800,
                        int(self.base_image.size[1] * 1800 / self.base_image.size[0]),
                    )
                )
            self.draw_base = ImageDraw.Draw(self.base_image)
        else:
            self.master.quit()

    def reset(self) -> None:
        """
        reset method resets the points list and labels, loads the current image,
        sets it as the base image, and enables the 'Done' button.

        Returns:
            None
        """
        self.points_list = []
        self.labels = []
        self.new_image(Image.open(self.images[self.image_num]))
        self.base_image = Image.open(self.images[self.image_num])
        self.draw_base = ImageDraw.Draw(self.base_image)
        self.done_button.config(state="disable")

    def previous(self) -> None:
        """
        previous method loads the previous image to be processed, resets the
        points list and labels, and sets it as the base image. If it is the
        first image, it displays a warning message to the user.

        Returns:
            None
        """
        self.points_list = []
        self.labels = []
        if self.image_num > 0:
            self.image_num -= 1
            self.new_image(Image.open(self.images[self.image_num]))
            self.base_image = Image.open(self.images[self.image_num])
            self.predictor.set_image(np.array(self.image))
            if self.base_image.size[0] > 1800:
                self.base_image = self.base_image.resize(
                    (
                        1800,
                        int(self.base_image.size[1] * 1800 / self.base_image.size[0]),
                    )
                )
            self.draw_base = ImageDraw.Draw(self.base_image)
        else:
            messagebox.showwarning(message="You are annotating the first image")

    def next(self) -> None:
        """
        next method loads the next image to be processed, resets the points list and
        labels, and sets it as the base image. If it is the last image, it displays
        a warning message to the user.

        Returns:
            None
        """
        self.points_list = []
        self.labels = []
        if self.image_num < len(self.images) - 1:
            self.image_num += 1
            self.new_image(Image.open(self.images[self.image_num]))
            self.base_image = Image.open(self.images[self.image_num])
            self.predictor.set_image(np.array(self.image))
            if self.base_image.size[0] > 1800:
                self.base_image = self.base_image.resize(
                    (
                        1800,
                        int(self.base_image.size[1] * 1800 / self.base_image.size[0]),
                    )
                )
            self.draw_base = ImageDraw.Draw(self.base_image)
        else:
            messagebox.showwarning(message="You are annotating the last image")

    def exit(self) -> None:
        """
        exit method prompts the user to confirm if they want to exit the application.
        If the user confirms, it quits the application. Otherwise, it does nothing.

        Returns:
            None
        """
        response = messagebox.askquestion("Exit", "Are you sure you want to exit?")
        if response == "yes":
            self.master.quit()
        else:
            pass


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


def main(args) -> None:
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
    main(get_args())
