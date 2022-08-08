# python train.py --config config/db_resnet50.yaml --use_dist False
file=train.py
config=config/master_lmdb.yaml
use_dist=False
python $file --config=$config --use_dist=$use_dist 
