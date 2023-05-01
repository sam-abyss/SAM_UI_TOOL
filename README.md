# SAM_UI_TOOL
UI tool for adding masks to images using Meta's Segment Anything Model

## Installation
To install, run the code below

`pip install "SAM_UI_TOOL @ git+https://github.com/sam-abyss/SAM_UI_TOOL@main"`

You will also need to download a [model checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints) in order to run the Segment Anything Model

Some people may have an issue with python being unable to find tkinter. If so, run the following installation:

`sudo apt-get install python3-tk`

## Running
To run this script, you will need three things:

- A text file listing the images to run the model on (see example.txt)
- A folder in which to save the masks
- A model checkpoint for the Segment Anything Model

Once you know the location of all these, you can run the script below replacing the placeholders with the locations

`SAM-ui.py --input text/file/location --output-dir mask/directory --model-path model/checkpoint/location`
