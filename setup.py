#!/usr/bin/env python3
"""
Install SAM_UI_TOOL library
"""

import os
from setuptools import setup

setup(
    name="SAM_UI_TOOL",
    description="UI tool for segmenting images with Metas Segment Anything Model.",
    packages=["SAM_UI_TOOL"],
    package_data={},
    install_requires=[
        "numpy",
        "onnxruntime",
        "opencv_contrib_python",
        "opencv-python",
        "Pillow",
        "pycocotools",
        "torch",
        "torchvision",
    ],
    scripts=["SAM_UI_TOOL/SAM-ui.py"],
)
