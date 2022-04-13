mkdir dataset/bkai
cd dataset/bkai
# gdown --id 1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml
# gdown --id 1AhEwdTOxByNiHLfZcxFm83ZtPivxGJlS
# gdown --id 1_Z4zY2Wk7vtxepUhUzttfddkM2b9cap2
# unzip vietnamese_original.zip
# unzip train_imgs.zip
# unzip train_gt.zip
cd ../../ultocr/loader/detection
python create_train_val_data.py
# cd ../../../
