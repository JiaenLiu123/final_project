# Author: Jiaen LIU
# Date: 2023-01-02
# This app is inspired by the following app:
#  https://huggingface.co/spaces/Theivaprakasham/layoutlmv2_sroie/tree/main
#  https://learnopencv.com/deep-learning-based-document-segmentation-using-semantic-segmentation-deeplabv3-on-custom-dataset/

# Import the required libraries
import os
import gc
import cv2
import numpy as np
import uuid
import pathlib
import streamlit as st
import re
from dateutil.parser import parse

from PIL import Image, ImageDraw, ImageFont
from layoutLM.layoutLMv3 import  get_labels, process_image
# from streamlit_drawable_canvas import st_canvas
from regex_script.test_all_regex import test_regex

import torch
import torchvision.transforms as torchvision_T
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large
from torchvision.datasets.utils import download_file_from_google_drive

from Semantic_Segmentation.seg import scan, image_preprocess_transforms
from utils.utils import resize_image, get_image_download_link, remove_shadows
from transformers import LayoutLMv2ForTokenClassification, LayoutLMv2FeatureExtractor, LayoutLMv2TokenizerFast
from layoutLM.layoutLMv3 import handle
from layoutLM.ocr import prepare_batch_for_inference

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.cache(allow_output_mutation=True)
def get_donut():
    from donut import DonutModel
    donut = DonutModel.from_pretrained("/home/jiaenliu/final_project/20230313_111731")
    return donut

@st.cache(allow_output_mutation=True)
def load_model(num_classes=2, model_name="mbv3", device= torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if model_name == "mbv3":
        model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True)
        checkpoint_path = os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C_aux_e3_pretrain.pth")
    else:
        model = deeplabv3_resnet50(num_classes=num_classes, aux_loss=True)
        checkpoint_path = os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")
    model.to(device)
    checkpoints = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()

    _ = model(torch.randn((1, 3, 384, 384)).to(device))
    return model

# Load LayoutLMv2 model
@st.cache(allow_output_mutation=True)
def get_layoutlmv2(ocr_lang = "fra"):
    feature_extractor = LayoutLMv2FeatureExtractor(ocr_lang=ocr_lang,tesseract_config="--psm 12 --oem 2")
    tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
    # processor = LayoutLMv2Processor(feature_extractor, tokenizer)
    model = LayoutLMv2ForTokenClassification.from_pretrained("Theivaprakasham/layoutlmv2-finetuned-sroie")
    return tokenizer, feature_extractor, model

# get the labels
@st.cache(allow_output_mutation=True)
def get_labels():
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
    width, height = image.size

    # encode the image, get the bounding boxes and the words
    encoding_feature_extractor = feature_extractor(image, return_tensors="pt")
    # print(encoding_feature_extractor.keys())
    # print(encoding_feature_extractor.words)
    # TODO: apply the regex to the words
    words, boxes = encoding_feature_extractor.words[0], encoding_feature_extractor.boxes[0]
    # print(words)
    text = " ".join(words)
    encoding = tokenizer(words, boxes=boxes, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    # encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')
    encoding["image"] = encoding_feature_extractor.pixel_values

    # forward pass
    outputs = model(**encoding)

    # get the text of OCR output
    

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    # print(true_predictions)
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]


    true_boxes = true_boxes[1:-1]
    true_predictions = true_predictions[1:-1]

    json_df = []

    for i,j in enumerate(true_predictions):
        if j != 'O' and j != 'o':
            json_dict = {}

            json_dict["TEXT"] =  words[i]
            json_dict["LABEL"] = j

            json_df.append(json_dict)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    # print(zip(true_predictions, true_boxes))
    for ix, (prediction, box) in enumerate(zip(true_predictions, true_boxes)):
        predicted_label = iob_to_label(prediction).lower()
        # if predicted_label != 'o':
        #     box_img = image.crop(box)
        #     text.append((pytesseract.image_to_string(box_img, lang='fra'), predicted_label))
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font)

    # get the text from the labeled box 
    # print(text)
    # date,total = test_regex(text)
    return image, text, json_df

# Check if the model is already downloaded
# If not, download the model
# Download trained models
if not os.path.exists(os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C_aux_e3_pretrain.pth")):
    # https://drive.google.com/file/d/17FwgKDD3pvcYjWSRs_GGOgG8NpTydLlr/view?usp=share_link
    print("Downloading Deeplabv3 with MobilenetV3-Large backbone...")
    download_file_from_google_drive(file_id=r"17FwgKDD3pvcYjWSRs_GGOgG8NpTydLlr", root=os.getcwd(), filename=r"model_mbv3_iou_mix_2C_aux_e3_pretrain.pth")

if not os.path.exists(os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")):
    print("Downloading Deeplabv3 with ResNet-50 backbone...")
    download_file_from_google_drive(file_id=r"1DEl6qLckFChSDlT_oLUbO2JpN776Qx-g", root=os.getcwd(), filename=r"model_r50_iou_mix_2C020.pth")

# Set some parameters for the app
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / "static"
DOWNLOADS_PATH = STREAMLIT_STATIC_PATH / "downloads"
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()

IMAGE_SIZE = 384
preprocess_transforms = image_preprocess_transforms()
sizes = (600,1200)
image = None
final = None
result = None
# global text

# Define the main function
def main():
    st.set_page_config(initial_sidebar_state="collapsed", layout="wide")
    st.title("Receipt Extractor: Semantic Segmentation using DeepLabV3-PyTorch, OCR using PyTesseract, LayoutLMv3, Donut and regex for key information extraction")
    uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

    method_seg = st.radio("Select Document Segmentation Model:", ("MobilenetV3-Large", "Resnet-50"), horizontal=True)
    method_du = st.radio("Select Document Understanding Model:", ("LayoutLMv3", "Donut", "LayoutLMv2"), horizontal=True)

    col1, col2, col3, col4 = st.columns(4)

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        # h, w = image.shape[:2]

        if method_seg == "MobilenetV3-Large":
            model = load_model(model_name="mbv3")
        else:
            model = load_model(model_name="r50")
        
        with col1:
            st.title("Input")
            st.image(image,channels="BGR", use_column_width=True)
        
        with col2:
            st.title("Scanned")
            final = scan(preprocess_transforms,image, model, IMAGE_SIZE)
            st.image(final, channels="BGR", use_column_width=True)
                        
        with col3:
            st.title("Preprocessed")
            scanned_image = remove_shadows(final)
            scanned_image = cv2.copyMakeBorder(scanned_image, 30, 30, 30, 30, cv2.BORDER_CONSTANT,value=[0,0,0])
            scanned_image = Image.fromarray(scanned_image)
            scanned_image = resize_image(scanned_image, sizes[0], sizes[1])
            st.image(scanned_image, channels="BGR", use_column_width=True)

        if final is not None:
            with col4:
                # annotated_img, text, json_df = process_image(scanned_image, feature_extractor, tokenizer, layoutlmv2, id2label, label2color)
                if method_du == "LayoutLMv3":
                    st.title("Annotated Image")
                    inference_bath = prepare_batch_for_inference([scanned_image])
                    context = {"model_dir": "Theivaprakasham/layoutlmv3-finetuned-sroie"}
                    json_df, annotated_img = handle(inference_bath, context)
                    text = inference_bath["text"][0]
                    st.image(annotated_img, channels="BGR", use_column_width=True)
                    # st.image(annotated_img, channels="BGR", use_column_width=True)
                elif method_du == "LayoutLMv2":
                    st.title("Annotated Image")
                    tokenizer, feature_extractor, layoutLMv2 = get_layoutlmv2()
                    id2label, label2color = get_labels()
                    annotated_img, text, json_df = process_image(scanned_image, feature_extractor,tokenizer, layoutLMv2, id2label, label2color)
                    st.image(annotated_img, channels="BGR", use_column_width=True)
                    # st.write("No annotated image available for LayoutLMv2")
                else:
                    donut = get_donut()
                    if torch.cuda.is_available():
                        model.half()
                        device = torch.device("cuda")
                        model.to(device)
                    else:
                        model.encoder.to(torch.bfloat16)
                    model.eval()
                    json_df = donut.inference(image=scanned_image, prompt="<s_sroie_donut>")
                    text = None
                    # st.write(json_df)
                    st.write("No annotated image available for Donut")
                
        
        # Display the output
        if text is not None:
            st.title("Extracted Text")
            st.write(text)
            st.title("Key Information extracted by regex")
            date, total = test_regex(text, True)
            st.write("Date: ", date[1].strftime("%d/%m/%Y %H:%M:%S"), f"({round(date[0] * 100)}% sure)")
            st.write("Total: ", total[1], f"({round(total[0] * 100)}% sure)")
        
        if json_df is not None:
            # print(json_df)
            # st.write(json_df)
            if method_du == "LayoutLMv3" :
                st.title("Key Information extracted by LayoutLMv3")
                date = []
                total = ""
                for entity in json_df["output"]:
                    if entity["label"] == "DATE" or entity["label"] == "date":
                        date.append(entity["text"])
                    elif entity["label"] == "TOTAL" or entity["label"] == "total":
                        total = entity["text"]
                
                st.write("Date: ", " ".join(date))
                st.write("Total: ", total)
                st.markdown(get_image_download_link(annotated_img, "output.png", "Download " + "Annotated image"), unsafe_allow_html=True)
            elif method_du == "LayoutLMv2":
                st.title("Key Information extracted by LayoutLMv2")
                for i in json_df:
                    if "TOTAL" in i['LABEL']:
                        st.write("Total: ", i['TEXT'])
                    elif "DATE" in i['LABEL']:
                        st.write("Date: ", i['TEXT'])
            else:
                st.title("Key Information extracted by Donut")
                st.write("Date: ", json_df["predictions"][0]["date"])
                st.write("Total: ", json_df["predictions"][0]["total"])

if __name__ == "__main__":
    main()

# {"prompt":"you will translate following sentences into chinese\n<ENGLISH SCRIPT>\n\n###\n\n", "completion":"<CHINESE SENTENCES FROM PREVIOUS WORK>"}