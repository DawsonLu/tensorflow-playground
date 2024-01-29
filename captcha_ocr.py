################################################################################
## OCR Model for Reading Captchas
## Based on tutorial:
## https://keras.io/examples/vision/captcha_ocr/
################################################################################

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
import keras
from keras import layers

# TF Testing
print(tf.__version__)

# Path to the data directory
data_path = "./dataset/captcha_dataset"
data_dir = Path(data_path)

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)