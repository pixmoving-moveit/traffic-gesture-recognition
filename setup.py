#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import sys
from setuptools import setup, find_packages

try:
    README = open('README.md').read()
except Exception:
    README = ""
VERSION = "0.0.1"

requirments = ["argparse", "numpy", "opencv-python", "tensorflow-gpu", "h5py", "scipy"]
# requirments = ["click", "boto3", "appdirs", "grpcio", "pyyaml", "six"]

# if sys.version_info.major < 3:
#     requirments += ["configparser", "pathlib"]

setup(
    name='traffic_gesture_recognition',
    version=VERSION,
    description='traffic_gesture_recognition',
    url="https://github.com/moveithackathon-canhack/traffic-gesture-recognition",
    long_description=README,
    author='Jay Young(yjmade), Adam Shan, Dean Zhang, Kunal Tyagi',
    author_email='carnd@yjmade.net',
    packages=find_packages(),
    install_requires=requirments,
    entry_points={
        'console_scripts': [
            # 'modelhub=modelhub.commands:main'
        ]
    },
)
