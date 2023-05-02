#!/usr/bin/env python3
import tkinter
from tkinter import ttk, messagebox
from typing import Tuple, List, Any
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import pathlib
import os

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
        self.next_button = ttk.Button(self.buttons, text="Next", command=self.next)
        self.next_button.grid(row=0, column=4)
        self.done_button = ttk.Button(
            self.buttons, text="Done", command=self.done, state="disabled"
        )
        self.done_button.grid(row=0, column=2)
        self.exit_button = ttk.Button(self.buttons, text="Exit", command=self.exit)
        self.exit_button.grid(row=0, column=20)
        self.buttons.grid(row=0, column=0)
        self.image_num = 0
        self.image = Image.open(self.images[self.image_num])
        self.orig_resolution = self.image.size
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
        self.label3 = tkinter.Label(
            master, text=f"{os.path.basename(self.images[self.image_num])}"
        )
        self.label3.grid(row=3, column=0)
        self.label.bind("<Button-1>", self.left_click)
        self.label.bind("<Button-3>", self.right_click)
        self.points_list = []
        self.labels = []
        self.maskImg = None
        self.sam_checkpoint = args.model_path
        self.out_dir = args.output_dir
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device="cuda" if args.cuda else "cpu")
        self.predictor = SamPredictor(self.sam)
        """
        The SAMForm class is a tkinter-based user interface for generating masks for images using SAM model.

        Args:
            args: an object containing arguments for the SAM model
            master: the tkinter master object for the GUI
            images: a list of image paths

        Attributes:
            master: A tkinter master object that represents the main window of the GUI.
            images: A list of file paths to images that the user wants to generate masks for.
            buttons: A tkinter frame that contains several buttons for interacting with the GUI.
            reset_button: A tkinter ttk.Button object that resets the GUI to its default state.
            submit_button: A tkinter ttk.Button object that generates a mask for the current image.
            previous_button: A tkinter ttk.Button object that loads the previous image in the images list.
            next_button: A tkinter ttk.Button object that loads the next image in the images list.
            done_button: A tkinter ttk.Button object that signals that the user has finished generating masks for the current image.
            exit_button: A tkinter ttk.Button object that exits the GUI.
            image_num: An integer that represents the index of the current image in the images list.
            image: A PIL Image object that represents the current image.
            orig_resolution: A tuple that contains the original resolution of the current image.
            draw: A PIL ImageDraw object that is used to draw on the image object.
            base_image: A PIL Image object that represents the original unmodified current image.
            draw_base: A PIL ImageDraw object that is used to draw on the base_image object.
            scan: A tkinter PhotoImage object that represents the current image and is displayed in the GUI.
            label: A tkinter Label object that displays the scan object in the GUI.
            label2: A tkinter Label object that displays the index of the current image in the images list.
            label3: A tkinter Label object that displays the file name of the current image.
            points_list: A list of tuples that contains the coordinates of points that the user has clicked on the image object.
            labels: A list of integers that represents the labels assigned to each point in points_list.
            maskImg: A PIL Image object that represents the mask generated for the current image.
            sam_checkpoint: A string that represents the file path to the checkpoint of the segmentation model.
            out_dir: A string that represents the file path to the directory where the generated masks will be saved.
            model_type: A string that represents the type of the segmentation model.
            sam: A segmentation model object that is used to generate masks for images.
            predictor: A SamPredictor object that is used to predict masks for images.

        Methods:
            __init__(self, args, master, images: List[str]) -> None: Initializes the SAMForm object with the given arguments.

            left_click(self, event):
                Event handler for left clicks on the scan image. Adds a point to the points_list and draws a circle on the image.

            right_click(self, event):
                Event handler for right clicks on the scan image. Removes the most recent point from the points_list and redraws the image.

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

    def position(self, event: tkinter.Event) -> Tuple[int, int]:
        x = event.x
        y = event.y
        return x, y

    def submit(self) -> None:
        mask, scores, logits = self.generate_masks()
        mask, _, _ = self.generate_masks(scores=scores, logits=logits)
        maskArr = np.array(mask[0], dtype=np.uint8) * 255
        self.maskImg = Image.fromarray(maskArr, mode="L")

        image2 = self.maskImg

        self.draw_mask(image2)

        self.done_button.config(state="normal")

    def left_click(self, event: tkinter.Event) -> None:
        x, y = self.position(event)
        # print(f"left click at {x},{y}")
        self.points_list.append((x, y))
        self.labels.append(1)
        self.draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill="green")
        self.draw_base.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill="green")
        self.new_image(self.image)
        # print(self.points_list, self.labels)

    def right_click(self, event: tkinter.Event) -> None:
        x, y = self.position(event)
        # print(f"right click at {x},{y}")
        self.points_list.append((x, y))
        self.labels.append(0)
        self.draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill="red")
        self.draw_base.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill="red")
        self.new_image(self.image)
        # print(self.points_list, self.labels)

    def generate_masks(self, logits=None, scores=None) -> Any:
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

        except TypeError:
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )

        return masks, scores, logits

    def draw_mask(self, mask) -> None:
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
        self.label3.grid_forget()
        self.label3 = tkinter.Label(
            self.master, text=f"{os.path.basename(self.images[self.image_num])}"
        )
        self.label3.grid(row=3, column=0)
        self.done_button.config(state="disable")

    def done(self) -> None:
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
        self.done_button.config(state="disable")

    def previous(self) -> None:
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

    def next(self) -> None:
        self.points_list = []
        self.labels = []
        if self.image_num < len(self.images) - 1:
            self.image_num += 1
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
            messagebox.showwarning(message="You are annotating the last image")

    def exit(self) -> None:
        response = messagebox.askquestion("Exit", "Are you sure you want to exit?")
        if response == "yes":
            self.master.quit()
        else:
            pass
