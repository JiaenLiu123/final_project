# final_project: Real-time Receipt Recognition

## Introduction

This repository contains the final project for JiaenLiu's Bachelor's degree in Computer Science at the Beijing Institute of Petrochemical Technology. The project is a web app that can recognize receipts in real time and extract the information from the receipts.

## Installation

Idealy, you need a machine in Ubuntu 18.04 to run this project. You can also run it in Windows, but you need to install tesseract OCR engine manually. The following packages are required to run this project:

```bash

# Make sure you have install gcc ≥ 5.4 and g++ ≥ 5.4, detectron2 requires them to compile the C++ code.
# If you don't have them, you can install them by:
sudo apt install gcc g++

# Install the required packages
python -m pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python -m pip install torchvision tesseract

# For Ubuntu to install Tesseract 5
sudo apt update
sudo add-apt-repository ppa:alex-p/tesseract-ocr-devel
sudo apt install -y tesseract-ocr
sudo apt update 
# Check the version is correct (Should be the latest version)
tesseract --version
# Be careful to make sure the fra.traineddata and eng.traineddata are correct

```

## Need to be improved
1. Correct the length of the prediction and the length of the words.  
While I try to get the output of the model, it seems like the lenght of prediction and lenght of words are not the same. There are no good solutions on the Internet. For my personal experience, I found if OCR is more accurate, the probility of this problem will be lower. So I think the model can be improved by using a better OCR model.



## Thanks
This project is based on the following projects:  
https://learnopencv.com/deep-learning-based-document-segmentation-using-semantic-segmentation-deeplabv3-on-custom-dataset/  
https://huggingface.co/spaces/Theivaprakasham/layoutlmv2_sroie  