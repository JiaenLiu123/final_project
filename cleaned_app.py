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
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large
from torchvision.datasets.utils import download_file_from_google_drive

from Semantic_Segmentation.seg import scan, image_preprocess_transforms
from utils.utils import resize_image, get_image_download_link, remove_shadows

from layoutLM.layoutLMv3 import handle
from layoutLM.ocr import prepare_batch_for_inference
@st.cache(allow_output_mutation=True)
def get_donut():
    from donut import DonutModel
    donut = DonutModel.from_pretrained("/home/jiaenliu/final_project/20230313_111731")
    return donut

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# Define the main function
def main():
    # st.set_page_config(layout="wide")
    st.set_page_config(initial_sidebar_state="collapsed", layout="wide")
    # tokenizer, feature_extractor, layoutlmv2 = get_layoutlmv3()

    # id2label, label2color = get_labels()
    st.title("Receipt Extractor: Semantic Segmentation using DeepLabV3-PyTorch, OCR using PyTesseract, LayoutLMv3 and regex for key information extraction")

    uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

    method = st.radio("Select Document Segmentation Model:", ("MobilenetV3-Large", "Resnet-50"), horizontal=True)
    
    col1, col2, col3, col4 = st.columns(4)
    method_seg = st.radio("Select Document Segmentation Model:", ("MobilenetV3-Large", "Resnet-50"), horizontal=True)

    method_du = st.radio("Select Document Understanding Model:", ("LayoutLMv3", "Donut"), horizontal=True)

    col1, col2, col3 = st.columns(3)

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
            st.title("preprocessed Image")
            scanned_image = remove_shadows(final)
            scanned_image = cv2.copyMakeBorder(scanned_image, 30, 30, 30, 30, cv2.BORDER_CONSTANT,value=[0,0,0])
            scanned_image = Image.fromarray(scanned_image)
            scanned_image = resize_image(scanned_image, sizes[0], sizes[1])
            st.image(scanned_image, channels="BGR", use_column_width=True)

        if final is not None:
            with col4:
                st.title("Annotated Image")
                # annotated_img, text, json_df = process_image(scanned_image, feature_extractor, tokenizer, layoutlmv2, id2label, label2color)
                inference_bath = prepare_batch_for_inference([scanned_image])
                context = {"model_dir": "Theivaprakasham/layoutlmv3-finetuned-sroie"}
                json_df, annotated_img = handle(inference_bath, context)
                text = inference_bath["text"][0]
                st.image(annotated_img, channels="BGR", use_column_width=True)
                if method_du == "LayoutLMv3":
                    annotated_img, text, json_df = process_image(scanned_image, feature_extractor, tokenizer, layoutlmv2, id2label, label2color)
                    st.image(annotated_img, channels="BGR", use_column_width=True)
                else:
                    donut = get_donut()
                    if torch.cuda.is_available():
                        model.half()
                        device = torch.device("cuda")
                        model.to(device)
                    else:
                        model.encoder.to(torch.bfloat16)
                    model.eval()
                    output = donut.inference(image=scanned_image, prompt="<s_sroie_donut>")
                    st.write(output)
                
        
        # Display the output
        if text is not None:
            st.title("Extracted Text")
            st.write(text)
            st.title("Key Information extracted by regex")
            date, total = test_regex(text, True)
            st.write("Date: ", date[1].strftime("%d/%m/%Y %H:%M:%S"), f"({round(date[0] * 100)}% sure)")
            st.write("Total: ", total[1], f"({round(total[0] * 100)}% sure)")
        
        if json_df is not None:
            st.title("Key Information extracted by LayoutLMv3")
            print(json_df)
            # st.write(json_df)
            date = []
            total = ""
            for entity in json_df["output"]:
                if entity["label"] == "DATE":
                    date.append(entity["text"])
                elif entity["label"] == "TOTAL":
                    total = entity["text"]
            
            st.write("Date: ", " ".join(date))
            st.write("Total: ", total)

        st.markdown(get_image_download_link(annotated_img, "output.png", "Download " + "Annotated image"), unsafe_allow_html=True)


if __name__ == "__main__":
    main()




# {"prompt":"you will translate following sentences into chinese\n<ENGLISH SCRIPT>\n\n###\n\n", "completion":"<CHINESE SENTENCES FROM PREVIOUS WORK>"}