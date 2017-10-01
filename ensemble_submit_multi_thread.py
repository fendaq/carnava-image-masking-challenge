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

#------------------------------multi_thread--------------------
# Time decorator
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r (%r, %r) %2.2f sec' % (method.__name__, args, kw, te-ts))
        return result
    return timed

# Create some helper functions
def get_time_left(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

# List of all images in the folder
def get_all_images(folder_path):
    # Get all the files
    features = sorted(os.listdir(folder_path))
    features_path =[]
    for iF in features:
        features_path.append(os.path.join(folder_path, iF))

    return features_path, [i_feature.split('.')[0] for i_feature in features]

def load_mask_image(mask_path):
    mask_image = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    prob = cv2.resize(mask_image,dsize=(CARVANA_WIDTH,CARVANA_HEIGHT),interpolation=cv2.INTER_LINEAR)
    mask_image = prob>127
    return mask_image

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs.tolist()

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def mask_to_row(mask_path):
    mask_name = "%s.jpg" % os.path.basename(mask_path).split('.')[0]
    return [mask_name, run_length_encode(load_mask_image(mask_path))]

@timeit
def create_submission(csv_file, inference_folder, num_workers=2, image_queue=4, dry_run=False):
    # Create file and writer, if a dry run is specified, we dont write anything
    if dry_run:
        writer_fcn = lambda x: x
    else:
        open_csv = open(csv_file, 'w')
        writer = csv.writer(open_csv, delimiter=',')
        writer_fcn = lambda x: writer.writerow(x)
        
    # Write the header
    writer_fcn(["img", "rle_mask"])
    
    # Wrapper for writing
    def writer_wrap(queue):
        while True:
            # Get stuff from queue
            x = queue.get(timeout=1)
            if x is None:
                break
            writer_fcn(x)
        return

    # wrapper for creating
    def rle_wrap(queues):
        while True:
            path = queues[0].get(timeout=1)
            if path is None:
                break
            if path == -1:
                queues[1].put(None)
                break
            this_str = mask_to_row(path)
            queues[1].put(this_str)
        return

    # Define the rle queue
    rle_queue = Queue(image_queue)
    # Allow a little bit more to be passed to the writer queue
    writer_queue = Queue(image_queue*2)
    
    # Define and start our workers
    rle_workers = num_workers
    rle_consumer = [Process(target=rle_wrap, args=([rle_queue, writer_queue],)) for _ in range(rle_workers)]
    csv_worker = Process(target=writer_wrap, args=(writer_queue,))
    for _p in rle_consumer:
        _p.start()
    csv_worker.start()

    # Fetch all images
    paths, names = get_all_images(inference_folder)

    # Now run through all the images
    sum_time = 0
    n_images = len(paths)
    for i, iMask in enumerate(paths):
        start_time = time.time()
        rle_queue.put(iMask)
        run_time = time.time() - start_time
        sum_time += run_time
        mean_time = sum_time / (i + 1)
        eta_time = mean_time * (n_images - i - 1)
        print("\r%d/%d: ETA: %s, AVE: %dms" % (i, n_images, get_time_left(eta_time), int(mean_time*1000)),\
                  end='',flush=True)
        
    # Poison pill
    for _ in range(num_workers-1):
        rle_queue.put(None)
    # Last worker will kill the writer 
    rle_queue.put(-1)
    
    # And join them
    for thread in rle_consumer:
        thread.join()
    csv_worker.join()
    
def run_submit2_multi_thread():
    start = timer()

    #final_out_dir = params.out_dir + params.ensemble_dir #+ '_post_train_no_src' # + '/post_train2_ensemble_source' 
    final_out_dir = params.out_dir + 'UNet1024_ASPP_08_ens2'

    mask_path =  final_out_dir+'/submit/test_mask' #mask_path
    #create_submission("", mask_path, dry_run=True)

    #TRAIN_IMG = "/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/ensemble/UNet1024_ASPP_08/submit/test_mask"
    #test_dir = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/ensemble/test/'
    #list_of_images = get_all_images(TRAIN_IMG)
    #print(list_of_images[1][1:4])

    #logging, etc --------------------
    os.makedirs(final_out_dir+'/submit/results',  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), final_out_dir +'/backup/submit.code.zip')
    
    log = Logger()
    log.open(final_out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')

    create_submission(final_out_dir + '/submit/temp_1.csv', mask_path, num_workers=3, image_queue=6)

    print('\n--delete last 1 commont--')
    # delete last 1 commont
    origin_f = open(final_out_dir +'/submit/temp_1.csv', 'r')
    new_f = open(final_out_dir + '/submit/temp_2.csv', 'w')
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
    df = pd.read_csv(final_out_dir + '/submit/temp_2.csv')

    dir_name = final_out_dir.split('/')[-1]
    gz_file  = final_out_dir + '/submit/results-%s.csv.gz'%dir_name
    df.to_csv(gz_file, index=False, compression='gzip')
    
    os.remove(final_out_dir +'/submit/temp_1.csv')
    os.remove(final_out_dir +'/submit/temp_2.csv')

    log.write('%0.2f min'%((timer()-start)/60))


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_submit2_multi_thread()
    
    print('\nsucess!')
