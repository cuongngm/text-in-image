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
from ultocr.utils.det_utils import test_preprocess

def convert_image(image_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = cv2.imread(image_path)
    h_origin, w_origin = img.shape[:2]
    tmp_img = test_preprocess(img, new_size=736, pad=False) 
    tmp_img = tmp_img.to(device)
    return tmp_img
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='Hyper_parameter')
    parser.add_argument('--path', type=str, default='assets/2.jpg', help='choose gpu device')
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
# model_name = 'dbnet'
# stage = 'Production'
# opt = parse_args()
# image_path = opt.path
# print(image_path)
# test_img = convert_image(image_path)



if __name__ == '__main__':
    import mlflow.pyfunc
    model_name = "dbnet"
    model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/Production")
    opt = parse_args()
    image_path = opt.path
    test_img = convert_image(image_path)
    out = model(test_img)
    print(out)
    """
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
    """
