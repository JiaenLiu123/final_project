import os
import cv2
import uuid
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re
import pandas as pd

import torch
import torchvision.transforms as torchvision_T
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast


def get_layoutlmv2(ocr_lang = "fra", tesseract_config = "--psm 12 --oem 2"):
    feature_extractor = LayoutLMv3FeatureExtractor(ocr_lang=ocr_lang,tesseract_config=tesseract_config)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base-uncased")
    # processor = LayoutLMv2Processor(feature_extractor, tokenizer)
    model = LayoutLMv3ForTokenClassification.from_pretrained("Theivaprakasham/layoutlmv3-finetuned-sroie")
    return tokenizer, feature_extractor, model

def get_labels():
    # dataset = load_dataset("darentang/sroie", split="test")
    # define id2label, label2color
    labels = ['O', 'B-COMPANY', 'I-COMPANY', 'B-DATE', 'I-DATE', 'B-ADDRESS', 'I-ADDRESS', 'B-TOTAL', 'I-TOTAL']
    id2label = {v: k for v, k in enumerate(labels)}
    label2color = {'B-ADDRESS': 'blue',
    'B-COMPANY': 'green',
    'B-DATE': 'red',
    'B-TOTAL': 'red',
    'I-ADDRESS': "blue",
    'I-COMPANY': 'green',
    'I-DATE': 'red',
    'I-TOTAL': 'red',
    'O': 'green'}

    label2color = dict((k.lower(), v.lower()) for k,v in label2color.items())
    return id2label, label2color

# Unnormalize the bounding box coordinates.
def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

# iob to label
def iob_to_label(label):
    return label

def process_image(image, feature_extractor, tokenizer, model, id2label, label2color):
    """
    Process the image and return predictions.

    Args:
        image (PIL.Image): Image to be processed.
        # processor (LayoutLMv2Processor): LayoutLMv2 processor.
        feature_extractor (LayoutLMv2FeatureExtractor): LayoutLMv2 feature extractor.
        tokenizer (LayoutLMv2TokenizerFast): LayoutLMv2 tokenizer.
        model (LayoutLMv2ForTokenClassification): LayoutLMv2 model.
        id2label (dict): Dictionary mapping label id to label name.
        label2color (dict): Dictionary mapping label name to color.
    Returns:
        PIL.Image: Image with predictions drawn on it.
    """
    if not os.path.exists("error_images"):
        os.mkdir("error_images")
    width, height = image.size

    # encode the image, get the bounding boxes and the words
    encoding_feature_extractor = feature_extractor(image, return_tensors="pt")
    # print(encoding_feature_extractor.keys())
    # print(encoding_feature_extractor.words)
    words, boxes = encoding_feature_extractor.words[0], encoding_feature_extractor.boxes[0]
    # print(words)
    text = " ".join(words)
    encoding = tokenizer(words, boxes=boxes, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    # encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')
    # encoding["image"] = encoding_feature_extractor.pixel_values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k, v in encoding.items():
        encoding[k] = v.to(device)

    model.to(device)

    # forward pass
    outputs = model(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

    true_predictions = true_predictions[1:-1]
    true_boxes = true_boxes[1:-1]

    # print(len(words),len(true_predictions),len(true_boxes))
    # if length of words, predictions and boxes are not equal, then save the image
    if len(words) != len(true_predictions) or len(words) != len(true_boxes):
        image.save("error_images/" +str(uuid.uuid4()) + ".jpg")
        print("There is an error when processing the image. Please check the error_images folder.")
    # print(words)
    # print(true_predictions)


    json_df = []

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    # print(zip(true_predictions, true_boxes))
    for ix, (prediction, box) in enumerate(zip(true_predictions, true_boxes)):
        predicted_label = iob_to_label(prediction).lower()
        if prediction != 'O' and prediction != 'o':
            json_dict = {}

            json_dict["TEXT"] =  words[ix]
            json_dict["LABEL"] = predicted_label
            
            json_df.append(json_dict)
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font)

    # get the text from the labeled box 
    # print(text)
    return image, text, json_df