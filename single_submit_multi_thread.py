import params

from common import *
from dataset.carvana_cars import *

from model.tool import *

import csv
import pandas as pd
import numpy as np
import os
import time
from multiprocessing import Process, Queue


    
def run_submit2_multi_thread():
    start = timer()

    out_dir = params.out_dir + params.save_path
    mask_path =  out_dir+'/submit/test_mask' #mask_path
    #create_submission("", mask_path, dry_run=True)

    #TRAIN_IMG = "/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/ensemble/UNet1024_ASPP_08/submit/test_mask"
    #test_dir = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/ensemble/test/'
    #list_of_images = get_all_images(TRAIN_IMG)
    #print(list_of_images[1][1:4])
    
    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')

    create_submission(out_dir + '/submit/temp_1.csv', mask_path, num_workers=4, image_queue=8)

    print('\n--delete last 1 commont--')
    # delete last 1 commont
    origin_f = open(out_dir +'/submit/temp_1.csv', 'r')
    new_f = open(out_dir + '/submit/temp_2.csv', 'w')
    reader = csv.reader(origin_f)
    writer = csv.writer(new_f)

    idx = 0
    for row in reader:
        #del row[-1]
        #print(row)

        idx += 1
        if idx>100065: break
        check = row[0]
        writer.writerow(row)

    assert (check!='img')
    
    origin_f.close()
    new_f.close()

    print('\n--convert to gzip--')

    #with gzip.GzipFile(filename='test.csv', mode='wb', compresslevel=9, fileobj='test.gz') as f:
    #    f.write(new_f)
    df = pd.read_csv(out_dir + '/submit/temp_2.csv')

    dir_name = out_dir.split('/')[-1]
    gz_file  = out_dir + '/submit/results-%s.csv.gz'%dir_name
    df.to_csv(gz_file, index=False, compression='gzip')
    
    os.remove(out_dir +'/submit/temp_1.csv')
    os.remove(out_dir +'/submit/temp_2.csv')

    log.write('%0.2f min'%((timer()-start)/60))


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_submit2_multi_thread()
    
    print('\nsucess!')
