# https://www.kaggle.com/adamhart/submission-file-generation-using-separate-threads?scriptVersionId=1501663
# https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/blob/master/test_submit_multithreaded.py


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