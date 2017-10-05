import os

import params
from dataset.carvana_cars import *

from masks2lmdb import *
from ensemble_multi_thread import *

split_file = CARVANA_DIR +'/split/'+ 'test_100064'
with open(split_file) as f:
    names = f.readlines()
names = [name.strip()for name in names]
num_test = len(names)

for i in range(7):
    start = timer()
    outputPath = params.out_dir + 'UNet1024_GCN_06_k%d/submit/test_lmdb'%(i+1)
    createDataset(outputPath, params.out_dir + 'UNet1024_GCN_06_k%d'%(i+1), names)
    print('total time: %.2f min' %((timer()-start)/60))

ensemble_png_multi_process()
