# https://discuss.pytorch.org/t/whats-the-best-way-to-load-large-data/2977
# https://github.com/meijieru/crnn.pytorch/blob/master/dataset.py#L60
# https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py

'''
class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)
'''

import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
#import base64

import params
from dataset.carvana_cars import *

import time


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def createDataset(outputPath, imagePath_, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    print('out_path = ' + imagePath_)
    nSamples = len(labelList)
    #os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = labelList[i]
        imagePath = imagePath_ + '/submit/test_mask/' + imagePath + '.png'

        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            #imageBin = base64.b64encode(f.read())
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        #imageKey = 'image-%09d' % cnt
        #labelKey = 'label-%09d' % cnt
        cache[label] = imageBin
        #cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('\rWritten %d / %d' % (cnt, nSamples),end='',flush=True)
        print('\rCurr ID %d / %d' % (cnt, nSamples),end='',flush=True)
        cnt += 1
    nSamples = cnt-1
    #cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('  Sucess Created dataset with %d samples' % nSamples)


if __name__ == '__main__':

    start = timer()

    outputPath = params.out_dir + params.save_path + '/submit/test_lmdb'
    #imagePathList = params.out_dir + params.save_path + '/submit/test_mask'

    split_file = CARVANA_DIR +'/split/'+ 'test_100064'
    with open(split_file) as f:
        names = f.readlines()
    names = [name.strip()for name in names]
    num_test = len(names)

    createDataset(outputPath, '', names)

    print('total time: %.2f min' %((timer()-start)/60))

    pass