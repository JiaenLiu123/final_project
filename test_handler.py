from layoutLM.layoutLMv3 import handle

from layoutLM.ocr import prepare_batch_for_inference
from asyncio.log import logger
import logging
import os

arr = []
batch_size = 32
# Create batches of size batch_size of the input data arr
def create_batches(arr, batch_size):
    for i in range(0, len(arr), batch_size):
        yield arr[i:i + batch_size]

if __name__ == "__main__":
    try:
        image_path = "/Users/liujiaen/Documents/Text_Recognition/dataset/findit/FindIt-Dataset-Train/T1-train/img"
        image_files = os.listdir(image_path)
        image_paths = [image_path + f"/{image_file}" for image_file in image_files if image_file != ".DS_Store"]
        # create_batches(arr, batch_size)
        for batch in create_batches(image_paths, batch_size):
            print(batch)
            inference_batch = prepare_batch_for_inference(batch)
            context = {"model_dir": "Theivaprakasham/layoutlmv3-finetuned-sroie"}
            handle(inference_batch, context)
        # inference_batch = prepare_batch_for_inference(image_paths)
        # context = {"model_dir": "Theivaprakasham/layoutlmv3-finetuned-sroie"}
        # handle(inference_batch, context)
    except Exception as e:
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(filename="logs/error_output.log", level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")
        logger = logging.getLogger(__name__)
        logger.error(e)