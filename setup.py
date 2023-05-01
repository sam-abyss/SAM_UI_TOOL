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
        "segment_anything",
        "torch",
        "torchvision",
    ],
    entry_points={
        "console_scripts": [
            "SAM-UI-TOOL=SAM_UI_TOOL.SAM-ui.py:main",
        ]
    },
)
