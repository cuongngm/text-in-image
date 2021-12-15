## ULTOCR
ULT OCR is an open-source toolbox based on Pytorch for text detection
and text recognition developed by UtmostLimit team. This project is
synthesis of our knowledge in the process of learning and understanding.
Welcome all contributions.


### Quickstart
```bash
pip install ultocr  # install our package

from ultocr.inference import End2end
model = End2end(img_path='./', det_model='DB', reg_model='MASTER')
result = model.get_result()
```


### Install
```bash
git clone https://github.com/cuongngm/text-in-image
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Train
Custom params in each config file of config folder then:
```bash
# For detection:
python train.py --config config/db_resnet50.yaml

# For recognition:
python train.py --config ./config/master.yaml
```
### Todo

### Reference
- [DB_text_minimal](https://github.com/huyhoang17/DB_text_minimal)
- [pytorchOCR](https://github.com/BADBADBADBOY/pytorchOCR)
- [MASTER-pytorch](https://github.com/wenwenyu/MASTER-pytorch)
