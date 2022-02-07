from fastapi import FastAPI, File, UploadFile, Request, Form
from typing import Optional
import io
import os
import time
from ultocr.inference import End2end


load_time = time.time()
model = End2end(img_path=None)
print('load model time', time.time() - load_time)
app = FastAPI()
@app.get('/')
async def read_root():
    return {'Hello': 'World'}

