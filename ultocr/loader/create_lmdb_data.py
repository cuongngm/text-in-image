import numpy as np
import sys
import os
import cv2
import lmdb
from tqdm import tqdm
from collections import Counter


def checkImageIsValid(imageBin):
    isvalid = True
    imgH = None
    imgW = None

    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    try:
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)

        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            isvalid = False
    except Exception as e:
        isvalid = False

    return isvalid, imgH, imgW


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def createDataset(outputPath, root_dir, annotation_path):
    """
    Create LMDB dataset for training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    word_fred = Counter()
    # annotation_path = os.path.join(root_dir, annotation_path)
    with open(annotation_path, 'r') as ann_file:
        lines = ann_file.readlines()
        annotations = [l.strip().split('\t') for l in lines]
        """ 
        annotations = []
        for l in lines:
            anns = l.strip().split(' ')
            image = anns[0]
            label = ' '.join(anns[1:])
            # for char in anns[1:]:
            #     word_fred.update([char])
            # label = label.replace('±', '+')
            # label = label.replace('•', '-')
            
            annotations.append([image, label])
        """
    # vocab = [w for w in word_fred.keys()]
    # print(vocab)
    nSamples = len(annotations)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 0
    error = 0

    pbar = tqdm(range(nSamples), ncols=100, desc='Create {}'.format(outputPath))
    for i in pbar:
        if len(annotations[i]) == 1:
            continue
        imageFile, label = annotations[i]
        # print('image', imageFile)
        # print('label', label)
        imagePath = os.path.join(root_dir, imageFile)

        if not os.path.exists(imagePath):
            error += 1
            continue

        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        isvalid, imgH, imgW = checkImageIsValid(imageBin)

        if not isvalid:
            error += 1
            continue
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt

        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        cnt += 1

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
        # if i == 20:
        #     break

    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)

    if error > 0:
        print('Remove {} invalid images'.format(error))
    print('Created dataset with %d samples' % nSamples)
    sys.stdout.flush()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='lmdb/new', help='output lmdb path')
    parser.add_argument('--root_dir', type=str, default='/data/cuongnm1/dataset', help='root dataset path')
    parser.add_argument('--annotation_path', type=str, default='/data/cuongnm1/dataset/label.txt', help='label path')
    args = parser.parse_args()
    createDataset(args.output_path, args.root_dir, args.annotation_path)