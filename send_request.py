import requests
import json
import os
import cv2
from PIL import Image
import torch
import mlflow
import numpy as np
import pandas as pd
import argparse

def convert_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 256))
    img = img.transpose(2, 1, 0)
    test_image = np.expand_dims(img,axis=0)
    print("SHAPE: ", test_image.shape)
    return test_image
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='Hyper_parameter')
    parser.add_argument('--path', type=str, default='saved/test_ocr/0.png', help='choose gpu device')
    args = parser.parse_args()
    return args
    

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# data = torch.randn(1, 3, 32, 256).to(device)
# data = np.random.randn(1, 256)
# data = pd.DataFrame(data)
# print(data)
# data_json = json.dumps(data.tolist())
# headers = {'Content-Type': 'application/json; format=pandas-records'}
# request_uri = 'http://172.26.33.199:2514/invocations'

# print(data.size())
model_name = 'master-model'
stage = 'Production'
opt = parse_args()
image_path = opt.path
print(image_path)
test_img = convert_image(image_path)



if __name__ == '__main__':
    opt = parse_args()
    image_path = opt.path
    print(image_path)
    test_img = convert_image(image_path)
    data = {"inputs": test_img.tolist()}
    data = json.dumps(data)
    response = requests.post(
        url="http://172.26.33.199:2514/invocations",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code != 200:
        raise Exception(
            "Status Code {status_code}. {text}".format(
                status_code=response.status_code, text=response.text
            )
        )

    print("Prediction: ", response.text)
