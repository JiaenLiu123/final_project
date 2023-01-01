# This file is written by Jiaen LIU
# Date: 2022-12-29

'''
    Target:
        1. Test layoutLMv2 on Custom French dataset.
        2. Fine tune layoutLMv2 on that dataset.
'''

# Now lets import the required libraries
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re
import pandas as pd

# For better visualization of the results
from alive_progress import alive_bar

from layoutLM.layoutLMv2 import get_layoutlmv2, get_labels, process_image

# Set some parameters
# Only use the true images
DATASET_PATH = "/Users/liujiaen/Documents/Text_Recognition/dataset/findit/FindIt-Dataset-Train/T1-train/img"

GRAND_TRUTH_PATH = "/Users/liujiaen/Documents/Text_Recognition/final_project/correct.csv"
RAMDOM_SEED = 42
IMAGE_SIZE = (600, 1200)

def resize_image(image, size_x, size_y):
    if image.size[0] > size_x and image.size[1] > size_y:
        image = image.resize((size_x, size_y))
    elif image.size[0] > size_x and image.size[1] < size_y:
        image = image.resize(size_x)
    elif image.size[1] > size_y and image.size[0] < size_x:
        image = image.resize(size_y)
    return image

# Load the model
tokenizer, feature_extractor, model = get_layoutlmv2(ocr_lang = "fra", tesseract_config = "--psm 12 --oem 2")
id2label, label2color = get_labels()


# Load the dataset
output = []
files = os.listdir(DATASET_PATH)
# with alive_bar(len(files)) as bar:
for file in files:
    if file.endswith(".jpg"):
        image = Image.open(os.path.join(DATASET_PATH, file))
        id = file.split(".")[0]
        image = resize_image(image, IMAGE_SIZE[0], IMAGE_SIZE[1])
        image, text, json_df = process_image(image, feature_extractor, tokenizer, model, id2label, label2color)
        json_df.append({"id": id})
        output.append(json_df)
        print(json_df)
            # bar()



# Save the output
predic_df = pd.DataFrame(output)
predic_df.to_csv("predic.csv", index=False)
