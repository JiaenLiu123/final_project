from regex_script.test_all_regex import test_regex
import pytesseract
import os
from tqdm import tqdm

from asyncio.log import logger
import logging
import traceback

def run_and_clean_tesseract_on_image(image_path):
    try:
        ocr_df = pytesseract.image_to_data(image_path, output_type="data.frame", lang="fra", config='--psm 12 --oem 2')
    except Exception as e:
        print(e)
        return None
    ocr_df = ocr_df.dropna()
    try:
        ocr_df = ocr_df.drop(ocr_df[ocr_df.text.str.strip() == ''].index)
    except Exception as e:
        print(e)
    # text_output = ' '.join(ocr_df.text.tolist())
    words = []
    for index, row in ocr_df.iterrows():
        word = {}
        origin_box = [row['left'], row['top'], row['left'] +
                    row['width'], row['top']+row['height']]
        word['word_text'] = row['text']
        word['word_box'] = origin_box
        words.append(word)
    text = " ".join([word["word_text"] for word in words])
    print(text)
    date, total = test_regex(text)
    print(date, total)
    image_name = os.path.basename(image_path)
    image_name = image_name[:image_name.find('.')]
    with open("text_output/regex.csv", "a") as f:
        f.write(f"{image_name};{date[1]};{total[1]} \n")
    return words


if __name__ == "__main__":
    try:
        image_path = "/Users/liujiaen/Documents/Text_Recognition/dataset/findit/FindIt-Dataset-Train/T1-train/img"
        image_files = os.listdir(image_path)
        # print(len(image_files))
        image_paths = [image_path + f"/{image_file}" for image_file in image_files if image_file != ".DS_Store"]
        # image_paths = image_paths[100:len(image_paths)]
        # create_batches(arr, batch_size)
        print(len(image_paths))
        index = 0
        for file in image_paths:
            print(index)
            print(file)
            run_and_clean_tesseract_on_image(file)
            index += 1
    except Exception as e:
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(filename="logs/error_output.log", level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")
        logger = logging.getLogger(__name__)
        logger.error(e)
        logger.error(traceback.format_exc())
        # logger.error(traceback.format_exc())
