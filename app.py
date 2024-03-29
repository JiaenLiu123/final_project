import os
import uuid
import gc
import io
import cv2
import base64
import pathlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from datasets import load_dataset
from streamlit_drawable_canvas import st_canvas
import pytesseract
from skimage.filters import threshold_local

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from regex_script.test_all_regex import test_regex

import time

import torch
import torchvision.transforms as torchvision_T
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large
from transformers import LayoutLMv2ForTokenClassification, LayoutLMv2FeatureExtractor, LayoutLMv2TokenizerFast
from torchvision.datasets.utils import download_file_from_google_drive

# 2022-12-01 add by @Jiaen LIU for LayoutLMv2
# Inspired by https://huggingface.co/spaces/Theivaprakasham/layoutlmv2_sroie/tree/main

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

# Process the image and return predictions.
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
    date,total = test_regex(text)
    return image, text, date,total, json_df



# Download trained models
if not os.path.exists(os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C_aux_e3_pretrain.pth")):
    # https://drive.google.com/file/d/17FwgKDD3pvcYjWSRs_GGOgG8NpTydLlr/view?usp=share_link
    print("Downloading Deeplabv3 with MobilenetV3-Large backbone...")
    download_file_from_google_drive(file_id=r"17FwgKDD3pvcYjWSRs_GGOgG8NpTydLlr", root=os.getcwd(), filename=r"model_mbv3_iou_mix_2C_aux_e3_pretrain.pth")

if not os.path.exists(os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")):
    print("Downloading Deeplabv3 with ResNet-50 backbone...")
    download_file_from_google_drive(file_id=r"1DEl6qLckFChSDlT_oLUbO2JpN776Qx-g", root=os.getcwd(), filename=r"model_r50_iou_mix_2C020.pth")



@st.cache(allow_output_mutation=True)
# add function to load model from google drive
def load_model(num_classes=2, model_name="mbv3", device= torch.device("c" if torch.cuda.is_available() else "cpu")):
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

def resize_image(image, size_x, size_y):
    if image.size[0] > size_x and image.size[1] > size_y:
        image = image.resize((size_x, size_y))
    elif image.size[0] > size_x and image.size[1] < size_y:
        image = image.resize(size_x)
    elif image.size[1] > size_y and image.size[0] < size_x:
        image = image.resize(size_y)
    return image


# def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
#     # initialize the dimensions of the image to be resized and
#     # grab the image size
#     dim = None
#     (h, w) = image.shape[:2]

#     # if both the width and height are None, then return the
#     # original image
#     if width is None and height is None:
#         return image

#     # check to see if the width is None
#     if width is None:
#         # calculate the ratio of the height and construct the
#         # dimensions
#         r = height / float(h)
#         dim = (int(w * r), height)

#     # otherwise, the height is None
#     else:
#         # calculate the ratio of the width and construct the
#         # dimensions
#         r = width / float(w)
#         dim = (width, int(h * r))

#     # resize the image
#     resized = cv2.resize(image, dim, interpolation = inter)

#     # return the resized image
#     return resized


# Add a scanner effect to the image.
def add_scanner_effect(img, parameters={
    "blur": (5,5),
    "blur_iterations": 1,
    "erode_kernel": np.ones((2,2), np.uint8),
    "erode_iterations": 1,
    "dilate_kernel": np.ones((2,2), np.uint8),
    "dilate_iterations": 1,
    "erode_kernel_2": np.ones((3,3), np.uint8),
    "erode_iterations_2": 1,
    "dilate_kernel_2": np.ones((3,3), np.uint8),
    "dilate_iterations_2": 1,
    "threshold_local_block_size": 21,
    "threshold_local_offset": 5,
    # "threshold_local_mode": 'reflect',
    "sharpen_kernel": np.array([[0,-1,0], [-1,9,-1], [0,-1,0]]),
}):
    # Convert the image to grayscale.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian blur to the image.
    img = cv2.GaussianBlur(img, parameters["blur"], parameters["blur_iterations"])

    img = cv2.erode(img, parameters["erode_kernel"], iterations=parameters["erode_iterations"])
    img = cv2.dilate(img, parameters["dilate_kernel"], iterations=parameters["dilate_iterations"])

    # Apply a threshold to the image.
    T = threshold_local(img, parameters["threshold_local_block_size"], offset=parameters["threshold_local_offset"], method="gaussian")
    img = (img > T).astype("uint8") * 255
    
    # img = cv2.erode(img, parameters["erode_kernel_2"], iterations=parameters["erode_iterations_2"])
    # img = cv2.dilate(img, parameters["dilate_kernel_2"], iterations=parameters["dilate_iterations_2"])

    # Sharpen
    img = cv2.filter2D(src=img, ddepth=-1, kernel=parameters["sharpen_kernel"])
    return img


def image_preprocess_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    # Preprocessing transforms. Convert to tensor and normalize.
    common_transforms = torchvision_T.Compose(
        [
            torchvision_T.ToTensor(),
            torchvision_T.Normalize(mean, std),
        ]
    )
    return common_transforms


def order_points(pts):
    """Rearrange coordinates to order:
    top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)

    # Sort by Y position (to get top-down)
    # pts = pts[np.argsort(pts[:, 1])]

    # rect[0] = pts[0] if pts[0][0] < pts[1][0] else pts[1]
    # rect[1] = pts[1] if pts[0][0] < pts[1][0] else pts[0]
    # rect[2] = pts[2] if pts[2][0] > pts[3][0] else pts[3]
    # rect[3] = pts[3] if pts[2][0] > pts[3][0] else pts[2]
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype("int").tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.

    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    # if maxWidth > maxHeight:
    #     destination_corners = [[0, 0], [maxHeight, 0], [maxHeight, maxWidth], [0, maxWidth]]

    
    return order_points(destination_corners)




def scan(image_true=None, trained_model=None, image_size=384, BUFFER=10, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Scan the image and return the scanned image
    Args:
        image_true (np.array): Image to be scanned
        trained_model (torch.nn.Module): Trained model
        image_size (int): Size of the image to be fed to the model
        BUFFER (int): Buffer to be added to the image
    Returns:
        scanned_image (np.array): Scanned image
    """
    global preprocess_transforms

    IMAGE_SIZE = image_size
    half = IMAGE_SIZE // 2

    imH, imW, C = image_true.shape

    # Resizing the image to the size of input to the model. (384, 384)
    image_model = cv2.resize(image_true, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE

    # Converting the image to tensor and normalizing it.
    image_model = preprocess_transforms(image_model)
    image_model = torch.unsqueeze(image_model, dim=0)

    # Sending the image to the device.
    image_model = image_model.to(device)

    with torch.no_grad():
        # Out: the output of the model
        out = trained_model(image_model)["out"].cpu()

    del image_model
    gc.collect()

    out = torch.argmax(out, dim=1, keepdims=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
    r_H, r_W = out.shape

    _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
    _out_extended[half : half + IMAGE_SIZE, half : half + IMAGE_SIZE] = out * 255
    out = _out_extended.copy()

    del _out_extended
    # Garbage collection
    gc.collect()

    # Edge Detection.
    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Finding the largest contour. (Assuming that the largest contour is the document)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # ==========================================
    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)

    corners = np.concatenate(corners).astype(np.float32)

    corners[:, 0] -= half
    corners[:, 1] -= half

    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    # check if corners are inside.
    # if not find smallest enclosing box, expand_image then extract document
    # else extract document

    if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (imW, imH))):

        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box = cv2.boxPoints(rect)
        box_corners = np.int32(box)
        #     box_corners = minimum_bounding_rectangle(corners)

        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])

        # Find corner point which doesn't satify the image constraint
        # and record the amount of shift required to make the box
        # corner satisfy the constraint
        if box_x_min <= 0:
            left_pad = abs(box_x_min) + BUFFER

        if box_x_max >= imW:
            right_pad = (box_x_max - imW) + BUFFER

        if box_y_min <= 0:
            top_pad = abs(box_y_min) + BUFFER

        if box_y_max >= imH:
            bottom_pad = (box_y_max - imH) + BUFFER

        # new image with additional zeros pixels
        image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, C), dtype=image_true.dtype)

        # adjust original image within the new 'image_extended'
        image_extended[top_pad : top_pad + imH, left_pad : left_pad + imW, :] = image_true
        image_extended = image_extended.astype(np.float32)

        # shifting 'box_corners' the required amount
        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        image_true = image_extended

    corners = sorted(corners.tolist())
    corners = order_points(corners)
    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

    final = cv2.warpPerspective(image_true, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LANCZOS4)
    final = np.clip(final, a_min=0, a_max=255)
    final = final.astype(np.uint8)

    return final


# Generating a link to download a particular image file.
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# OCR the document by using the pytesseract library.
def ocr_document(image, lang="fra", width=600):
    imW = image.shape[1] # Get the size of the image
    if imW > width:
        # Resize the image to the width of 600 if the width is greater than 600 do that.
        image = image_resize(image, width=width)
    # Add a scanner effect to the image
    
    start_time = time.time()
    # image = add_scanner_effect(image)
    end_time = time.time()
    print("Scanner effect time: ", end_time - start_time)

    start_time = time.time()
    # OCR the image using the pytesseract library.

    data = pytesseract.image_to_data(image, lang=lang,output_type="dict")
    words = data["text"]

    # remove empty strings
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]

    words = " ".join(words)


    # end_time = time.time()

    # print("Tesseract time: ", end_time - start_time)

    return words,image

def remove_shadows(image):
    # convert the image to grayscale and blur it
    rgb_planes = cv2.split(image)
    # result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        
    # result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm

# TODO: Redesign the UI of the application.
# We create a downloads directory within the streamlit static asset directory
# and we write output files to it
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / "static"
DOWNLOADS_PATH = STREAMLIT_STATIC_PATH / "downloads"
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()

IMAGE_SIZE = 384
preprocess_transforms = image_preprocess_transforms()
image = None
final = None
result = None

st.set_page_config(initial_sidebar_state="collapsed")

tokenizer, feature_extractor, layoutLMv2 = get_layoutlmv2()
id2label, label2color = get_labels()

st.title("Document Scanner: Semantic Segmentation using DeepLabV3-PyTorch, OCR using PyTesseract and LayoutLMv2 for NER")

uploaded_file = st.file_uploader("Upload Document Image :", type=["png", "jpg", "jpeg"])

method = st.radio("Select Document Segmentation Model:", ("MobilenetV3-Large", "Resnet-50"), horizontal=True)

col1, col2,col3 = st.columns((6, 5,5))

if uploaded_file is not None:

    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    h, w = image.shape[:2]

    if method == "MobilenetV3-Large":
        model = load_model(model_name="mbv3")
    else:
        model = load_model(model_name="r50")

    with col1:
        st.title("Input")
        st.image(image, channels="BGR", use_column_width=True)

    with col2:
        st.title("Scanned")
        final = scan(image_true=image, trained_model=model, image_size=IMAGE_SIZE)
        st.image(final, channels="BGR", use_column_width=True)

    if final is not None:
        with col3:
            # OCR the document
            # print(type(final))
            scanned_img = remove_shadows(final)
            scanned_img = Image.fromarray(scanned_img)
            print(type(scanned_img))
            print(scanned_img.size[0], scanned_img.size[1])
            scanned_img = resize_image(scanned_img, size_x=600, size_y=1200)
            # print(type(scanned_img))

            # text,scanned_img = ocr_document(scanned_img)
            # text = pytesseract.image_to_string(scanned_img, lang="fra")
            # st.title("ORG OCR Output")
            # st.write(text)
            # st.title("Scanned Image")
            st.image(scanned_img, use_column_width=True)

            annotated_img, text, date, total,json_df = process_image(scanned_img, feature_extractor,tokenizer, layoutLMv2, id2label, label2color)
            st.image(annotated_img, use_column_width=True)
            st.title("LayoutLMv2 Output")
            for i in json_df:
                if "TOTAL" in i['LABEL']:
                    st.write("Total: ", i['TEXT'])
                elif "DATE" in i['LABEL']:
                    st.write("Date: ", i['TEXT'])
            # st.write(json_df)
            st.title("Regex Output")
            st.write("Date: ", date[1].strftime("%d/%m/%Y %H:%M:%S"), f"({round(date[0] * 100)}% sure)")
            st.write("Total: ", total[1], f"({round(total[0] * 100)}% sure)")
            st.title("OCR Output")
            st.write(text)
            # Display link.
            result = Image.fromarray(final[:, :, ::-1])
            st.markdown(get_image_download_link(result, "output.png", "Download " + "Output"), unsafe_allow_html=True)


# if not os.path.exists(os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C_aux_e3_pretrain.pth")):

# Create a output_img directory if it doesn't exist
if not os.path.exists(os.path.join(os.getcwd(), "output_img")):
    os.makedirs(os.path.join(os.getcwd(), "output_img"))

# save the output image and regex output to the output_img directory
if result is not None and json_df is not None:
    st.write("If the result is correct, click the button to save the output image and regex output")
    if st.button("Save Output"):
        name = str(uuid.uuid4())
        # Save the output image to the output_img directory
        img_str = name + ".png"
        result.save(os.path.join(os.getcwd(), "output_img") +  "/" + img_str, format="PNG")
        st.write("Output image saved to output_img directory")
        # Save the regex output to a text file
        text_str = name + ".txt"
        with open(os.path.join(os.getcwd(), "output_img") +  "/" + text_str , "w") as f:
            # f.write(json.dumps(json_df, indent=4))
            f.write("Date , " + date[1].strftime("%d/%m/%Y") + "\n")
            f.write("Total , " + str(total[1]) + "")
        st.write("Regex output saved to output_img directory")
    # result.save(os.path.join(os.getcwd(), "output_img") / "output.png", format="PNG")
