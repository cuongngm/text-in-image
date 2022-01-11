## ULTOCR
ULT OCR is an open-source toolbox based on Pytorch for text detection
and text recognition. This project is synthesis of our knowledge in the process of learning and understanding.
Welcome all contributions.


### Quickstart
```bash
pip install ultocr  # install our project with package
# for inference phase
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
### Prepare data

### Train
Custom params in each config file of config folder then:

Single gpu training:
```bash
python train.py --config config/db_resnet50.yaml --use_dist False
```
Multi gpu training:
```bash
# assume we have 2 gpu
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr=127.0.0.1 --master_post=5555 train.py --config config/db_resnet50.yaml
```
### Todo
- [x] Multi gpu training
- [ ] Release model zoo
- [ ] Pytorch lightning
- [ ] Tracking experiments with Mlflow
- [ ] Model serving with Mlflow
- [ ] Key information extraction
- [ ] Image orientation classifier
- [ ] Add more text detection and recognition model

### Reference
- [DB_text_minimal](https://github.com/huyhoang17/DB_text_minimal)
- [pytorchOCR](https://github.com/BADBADBADBOY/pytorchOCR)
- [MASTER-pytorch](https://github.com/wenwenyu/MASTER-pytorch)
- [DBNet.pytorch](https://github.com/WenmuZhou/DBNet.pytorch)
