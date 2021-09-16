# text-in-image
Refactor code text detection + text recognition
# Install
```bash
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install ultocr
```
# Train
Custom params in each config file of config folder then:
```bash
# For detection:
python tools/det_train.py --config ./config/db_resnet50.yaml

# For recognition:
python tools/reg_train.py --config ./config/master.yaml
```

# Inference
```bash
# For detection:
python tools/det_infer.py

# For recognition:
python tools/reg_infer.py
```

# Quickstart
```bash
from ultocr.inference import End2end
model = End2end(img_path='./', det_model='DB', reg_model='MASTER')
result = model.get_result()
```


