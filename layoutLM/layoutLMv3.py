# Description: This file contains the code for the layoutLMv3 model handler.

from .utils import load_model,load_processor,normalize_box,compare_boxes,adjacent
from .annotate_image import get_flattened_output,annotate_image
from PIL import Image,ImageDraw, ImageFont
import logging
import torch
import json
import os
import cv2
import uuid
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re
import pandas as pd

arr = []
batch_size = 32
# Create batches of size batch_size of the input data arr
def create_batches(arr, batch_size):
    for i in range(0, len(arr), batch_size):
        yield arr[i:i + batch_size]

logger = logging.getLogger(__name__)

class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.model = None
        self.model_dir = None
        self.device = 'cpu'
        self.error = None
        # self._context = None
        # self._batch_size = 0
        self.initialized = False
        self._raw_input_data = None
        self._processed_data = None
        self._images_size = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        logger.info("Loading transformer model")

        self._context = context
        properties = self._context
        # self._batch_size = properties["batch_size"] or 1
        self.model_dir = properties.get("model_dir")
        self.model = self.load(self.model_dir)
        self.initialized = True

    def preprocess(self, batch):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready
        # assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))
        inference_dict = batch
        self._raw_input_data = inference_dict
        processor = load_processor()
        images = [Image.open(path).convert("RGB")
                  for path in inference_dict['image_path']]
        self._images_size = [img.size for img in images]
        words = inference_dict['words']
        boxes = [[normalize_box(box, images[i].size[0], images[i].size[1])
                  for box in doc] for i, doc in enumerate(inference_dict['bboxes'])]
        encoded_inputs = processor(
            images, words, boxes=boxes, return_tensors="pt", padding="max_length", truncation=True)
        self._processed_data = encoded_inputs
        return encoded_inputs

    def load(self, model_dir):
        """The load handler is responsible for loading the hunggingface transformer model.
        Returns:
            hf_pipeline (Pipeline): A Hugging Face Transformer pipeline.
        """
        # TODO model dir should be microsoft/layoutlmv2-base-uncased
        model = load_model(model_dir)
        return model

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # TODO load the model state_dict before running the inference
        # Do some inference call to engine here and return output
        with torch.no_grad():
            inference_outputs = self.model(**model_input)
            predictions = inference_outputs.logits.argmax(-1).tolist()
        results = []
        for i in range(len(predictions)):
            tmp = dict()
            tmp[f'output_{i}'] = predictions[i]
            results.append(tmp)

        return [results]

    def postprocess(self, inference_output):
        docs = []
        k = 0
        for page, doc_words in enumerate(self._raw_input_data['words']):
            doc_list = []
            width, height = self._images_size[page]
            for i, doc_word in enumerate(doc_words, start=0):
                word_tagging = None
                word_labels = []
                word = dict()
                word['id'] = k
                k += 1
                word['text'] = doc_word
                word['pageNum'] = page + 1
                word['box'] = self._raw_input_data['bboxes'][page][i]
                _normalized_box = normalize_box(
                    self._raw_input_data['bboxes'][page][i], width, height)
                for j, box in enumerate(self._processed_data['bbox'].tolist()[page]):
                    if compare_boxes(box, _normalized_box):
                        if self.model.config.id2label[inference_output[0][page][f'output_{page}'][j]] != 'O':
                            word_labels.append(
                                self.model.config.id2label[inference_output[0][page][f'output_{page}'][j]][2:])
                        else:
                            word_labels.append('other')
                if word_labels != []:
                    word_tagging = word_labels[0] if word_labels[0] != 'other' else word_labels[-1]
                else:
                    word_tagging = 'other'
                word['label'] = word_tagging
                word['pageSize'] = {'width': width, 'height': height}
                if word['label'] != 'other':
                    doc_list.append(word)
            spans = []
            def adjacents(entity): return [
                adj for adj in doc_list if adjacent(entity, adj)]
            output_test_tmp = doc_list[:]
            for entity in doc_list:
                if adjacents(entity) == []:
                    spans.append([entity])
                    output_test_tmp.remove(entity)

            while output_test_tmp != []:
                span = [output_test_tmp[0]]
                output_test_tmp = output_test_tmp[1:]
                while output_test_tmp != [] and adjacent(span[-1], output_test_tmp[0]):
                    span.append(output_test_tmp[0])
                    output_test_tmp.remove(output_test_tmp[0])
                spans.append(span)

            output_spans = []
            for span in spans:
                if len(span) == 1:
                    output_span = {"text": span[0]['text'],
                                   "label": span[0]['label'],
                                   "words": [{
                                       'id': span[0]['id'],
                                       'box': span[0]['box'],
                                       'text': span[0]['text']
                                   }],
                                   }
                else:
                    output_span = {"text": ' '.join([entity['text'] for entity in span]),
                                   "label": span[0]['label'],
                                   "words": [{
                                       'id': entity['id'],
                                       'box': entity['box'],
                                       'text': entity['text']
                                   } for entity in span]

                                   }
                output_spans.append(output_span)
            docs.append({f'output': output_spans})
        return [json.dumps(docs, ensure_ascii=False)]

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        model_input = self.preprocess(data)
        # print()
        model_out = self.inference(model_input)
        inference_out = self.postprocess(model_out)[0]
        with open(f"LayoutlMV3InferenceOutput.json", 'w') as inf_out:
            inf_out.write(inference_out)
        inference_out_list = json.loads(inference_out)
        flattened_output_list = get_flattened_output(inference_out_list)
        for i, flattened_output in enumerate(flattened_output_list):
            annotate_image(data['image_path'][i], flattened_output)
            image_name = os.path.basename(data['image_path'][i])
            image_name = image_name[:image_name.find('.')]
            # print(flattened_output)
            with open("text_output/output.csv", "a+") as f:
                for entity in flattened_output["output"]:
                    print(entity)
                    if entity['label'] == 'TOTAL' or entity['label'] == 'DATE':
                        f.write(f"{image_name},{entity['text']},{entity['label']}\n")
            


_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None
    
    # print("executed handle")

    return _service.handle(data, context)



# class LayoutLMv3Interface:

#     def __init__(self, ocr_lang="fra", tesseract_config="--psm 12 --oem 2"):
#         self.model = None
#         self.model_path = None
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.error = None
#         self.initialized = False
#         self._raw_input_data = None
#         self._processed_data = None
#         self._images_size = None
    
#     def initialize(self, context):
#         self._context = context
#         properties = self._context
        
#         self.initialized = True


def get_layoutlmv3(ocr_lang = "fra", tesseract_config = "--psm 12 --oem 2"):
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
        print(len(words),len(true_predictions),len(true_boxes))
        print("There is an error when processing the image. Please check the error_images folder.")
    # print(words)
    # print(true_predictions)


    json_df = []

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    # print(zip(true_predictions, true_boxes))
    for ix, (prediction, box) in enumerate(zip(true_predictions, true_boxes)):
        predicted_label = iob_to_label(prediction).lower()
        if prediction != 'O':
            json_dict = {}

            json_dict["TEXT"] =  words[ix]
            json_dict["LABEL"] = predicted_label
            
            json_df.append(json_dict)
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font)

    # get the text from the labeled box 
    # print(text)
    return image, text, json_df