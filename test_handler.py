from layoutLM.layoutLMv3 import handle

from layoutLM.ocr import prepare_batch_for_inference
from asyncio.log import logger
import logging
import os
import traceback
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image,ImageDraw, ImageFont
from matplotlib.pyplot import imshow
import numpy as np

if not os.path.exists("output_img"):
    os.makedirs("output_img")

# arr = []
batch_size = 10
# Create batches of size batch_size of the input data arr
def create_batches(arr, batch_size):
    for i in range(0, len(arr), batch_size):
        yield arr[i:i + batch_size]

if __name__ == "__main__":
    try:
        # image_path = "/home/student/T1-train/img"
        # image_files = os.listdir(image_path)
        # print(len(image_files))
        # image_paths = [image_path + f"/{image_file}" for image_file in image_files if image_file != ".DS_Store"]
        # # image_paths = image_paths[100:len(image_paths)]
        # # create_batches(arr, batch_size)
        # for batch in create_batches(image_paths, batch_size):
        # print(batch)
        test_image = [Image.open("/home/jiaenliu/final_project/testimage/4.jpg").convert("RGB")]
        inference_batch = prepare_batch_for_inference(test_image)
        context = {"model_dir": "Theivaprakasham/layoutlmv3-finetuned-sroie"}
        data, img = handle(inference_batch, context)
        print(data)
        imshow(np.asarray(img))

    except Exception as e:
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(filename="logs/error_output.log", level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")
        logger = logging.getLogger(__name__)
        logger.error(e)
        logger.error(traceback.format_exc())
        # logger.error(traceback.format_exc())