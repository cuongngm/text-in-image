import os
import multiprocessing
from pathlib import Path
from base import crop_img
from create_lmdb_data import createDataset


def create_detect_train_data(root='../../dataset/bkai/vietnamese/'):
    Path(root + 'images').mkdir(parents=True, exist_ok=True)
    for imgname in os.listdir(os.path.join(root, 'train_images')):
        os.rename(os.path.join(root, 'train_images', imgname), os.path.join(root, 'images', imgname[2:]))
    for imgname in os.listdir(os.path.join(root, 'test_image')):
        os.rename(os.path.join(root, 'test_image', imgname), os.path.join(root, 'images', imgname[2:]))
    for imgname in os.listdir(os.path.join(root, 'unseen_test_images')):
        os.rename(os.path.join(root, 'unseen_test_images', imgname), os.path.join(root, 'images', imgname[2:]))
    for labelname in os.listdir(os.path.join(root, 'labels')):
        label_id = labelname[3:-4]
        label_id = label_id.zfill(4)
        os.rename(os.path.join(root, 'labels', labelname), os.path.join(root, 'labels', 'gt_' + label_id + '.txt'))


def create_recog_train_data(root='../../dataset/bkai/'):
    Path(root + 'recog').mkdir(parents=True, exist_ok=True)
    Path(root + 'recog/data_ver2').mkdir(parents=True, exist_ok=True)
    Path(root + 'recog/data_ver1').mkdir(parents=True, exist_ok=True)
    """
    nproc = 8
    p = multiprocessing.Pool(nproc)
    for i in range(nproc):
        p.apply_async(crop_img, (os.path.join(root, 'vietnamese/images'), os.path.join(root, 'vietnamese/labels'), os.path.join(root, 'recog/data_ver1'), os.path.join(root, 'recog', 'data_ver1.txt',))
        p.apply_async(crop_img, (os.path.join(root, 'training_img'), os.path.join(root, 'training_gt'), os.path.join(root, 'recog/data_ver2'), os.path.join(root, 'recog', 'data_ver2.txt',))
    p.close()
    p.join()
    """
    crop_img(img_dir=os.path.join(root, 'vietnamese/images'), box_dir=os.path.join(root, 'vietnamese/labels'), save_dir=os.path.join(root, 'recog/data_ver1'), save_label_file=os.path.join(root, 'recog', 'data_ver1.txt'))
    crop_img(img_dir=os.path.join(root, 'training_img'), box_dir=os.path.join(root, 'training_gt'), save_dir=os.path.join(root, 'recog/data_ver2'), save_label_file=os.path.join(root, 'recog', 'data_ver2.txt'))
    createDataset(os.path.join(root, 'recog/train_lmdb'), os.path.join(root, 'recog'), os.path.join(root, 'recog/data_ver1.txt'))
    createDataset(os.path.join(root, 'recog/test_lmdb'), os.path.join(root, 'recog'), os.path.join(root, 'recog/data_ver2.txt'))
    
if __name__ == '__main__':
    create_recog_train_data()
