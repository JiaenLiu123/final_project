from layoutLM.layoutLMv3 import handle

from layoutLM.ocr import prepare_batch_for_inference
from asyncio.log import logger
import logging
import os

if __name__ == "__main__":
    # try:
        image_path = "/Users/liujiaen/Documents/Text_Recognition/final_project/testimage"
        image_files = os.listdir(image_path)
        image_paths = [image_path + f"/{image_file}" for image_file in image_files if image_file != ".DS_Store"]
        inference_batch = prepare_batch_for_inference(image_paths)
        context = {"model_dir": "Theivaprakasham/layoutlmv3-finetuned-sroie"}
        handle(inference_batch, context)
    # except Exception as e:
    #     os.makedirs("logs", exist_ok=True)
    #     logging.basicConfig(filename="logs/error_output.log", level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")
    #     logger = logging.getLogger(__name__)
    #     logger.error(e)