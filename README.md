# final_project: Real-time Receipt Recognition

## Introduction

This repository contains the final project for JiaenLiu's Bachelor's degree in Computer Science at the Beijing Institute of Petrochemical Technology. The project is a web app that can recognize receipts in real time and extract the information from the receipts.

## Installation

Idealy, you need a machine in Ubuntu 18.04 to run this project. You can also run it in Windows, but you need to install tesseract OCR engine manually. The following packages are required to run this project:

```bash
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





## Thanks
This project is based on the following projects:  
https://learnopencv.com/deep-learning-based-document-segmentation-using-semantic-segmentation-deeplabv3-on-custom-dataset/  
https://huggingface.co/spaces/Theivaprakasham/layoutlmv2_sroie  