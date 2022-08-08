from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import Optional
from PIL import Image
import requests
import uvicorn
import io
import os
import time
import argparse
from pydantic import BaseModel
from ultocr.inference import OCR


def download_image(image_url):
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content)).convert('RGB')
    return image


class Result(BaseModel):
    url: str
    status_code: str
    description: Optional[str] = None
    text: Optional[str] = None
    latency: Optional[str] = None


def return_response(response):
    json_compatible_response_data = jsonable_encoder(response)
    return JSONResponse(json_compatible_response_data)


def parse_args():
    parser = argparse.ArgumentParser(description='Hyper parameter')
    parser.add_argument('--det_model', type=str, default='DB', help='text detection model')
    parser.add_argument('--reg_model', type=str, default='MASTER', help='text recognition model')
    parser.add_argument('--det_config', type=str, default='config/db_resnet50.yaml', help='DBnet config')
    parser.add_argument('--reg_config', type=str, default='config/master.yaml', help='MASTER config')
    parser.add_argument('--det_weight', type=str, default='saved/db_pretrain.pth', help='DBnet weight')
    parser.add_argument('--reg_weight', type=str, default='saved/master_pretrain.pth', help='MASTER weight')
    args = parser.parse_args()
    return args


app = FastAPI()


@app.get('/predict')
def predict(image_url: str):
    try:
        img = download_image(image_url)
    except Exception:
        result = Result(url=image_url, status_code=400, description='Cant download image..')
        response = return_response(result)
        return response
    start = time.time()
    text = model.get_result(img)
    text = '\n'.join(text)
    end = time.time() - start
    result = Result(url=image_url, status_code=200, latency=end, description='Get text sucessfull..', text=text)
    response = return_response(result)
    return response


if __name__ == '__main__':
    opt = parse_args()
    load_time = time.time()
    model = OCR(opt.det_model, opt.reg_model, opt.det_config, opt.reg_config, opt.det_weight, opt.reg_weight)
    print('load model time', time.time() - load_time)
    
    uvicorn.run(app, port=8000, host="127.0.0.1", reload=False)
