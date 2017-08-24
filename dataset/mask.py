from common import *

from dataset.carvana_cars import *
from dataset.tool import *



#---------------------------------------------------------------------------------
# https://www.kaggle.com/tunguz/baseline-2-optimal-mask
# https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation
# submission in "run-length encoding"
def run_length_encode0(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = np.where(mask.flatten()==1)[0]
    inds = inds + 1  # one-indexed : https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37108
    runs = []
    prev = -1
    for i in inds:
        if (i > prev + 1): runs.extend((i, 0)) #They are one-indexed
        runs[-1] += 1
        prev = i

    rle = ' '.join([str(r) for r in runs])
    return rle

#https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

def run_length_decode(rel, H, W, fill_value=255):
    mask = np.zeros((H*W),np.uint8)
    rel  = np.array([int(s) for s in rel.split(' ')]).reshape(-1,2)
    for r in rel:
        start = r[0]
        end   = start +r[1]
        mask[start:end]=fill_value
    mask = mask.reshape(H,W)
    return mask



#check rle
def check_rle():

    if 0: #check one mask file
        #opencv does not read gif

        mask_file = '/root/share/[data]/kaggle-carvana-cars-2017/annotations/train_masks/0cdf5b5d0ce1_01_mask.gif'
        mask = PIL.Image.open(mask_file)  #cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = np.array(mask)

        #im_show('mask', mask*255, resize=0.25)
        #cv2.waitKey(0)
        mask1 = cv2.resize(mask,(0,0), fx=0.25, fy=0.25)
        im_show('mask1', mask1*255, resize=1)
        rle = run_length_encode(mask1)

        cv2.waitKey(0)



    if 1: #check with train_masks.csv given

        csv_file  = CARVANA_DIR + '/masks_train.csv'  # read all annotations
        mask_dir  = CARVANA_DIR + '/annotations/train'  # read all annotations
        df  = pd.read_csv(csv_file)
        for n in range(10):
            shortname = df.values[n][0].replace('.jpg','')
            rle_hat   = df.values[n][1]

            mask_file = mask_dir + '/' + shortname + '_mask.gif'
            mask = PIL.Image.open(mask_file)
            mask = np.array(mask)
            rle  = run_length_encode(mask)
            #im_show('mask', mask*255, resize=0.25)
            #cv2.waitKey(0)
            match = rle == rle_hat
            print('%d match=%s'%(n,match))



#check rle
def check_rle_decode():
        csv_file  = CARVANA_DIR + '/masks_train.csv'  # read all annotations
        mask_dir  = CARVANA_DIR + '/annotations/train'  # read all annotations
        df  = pd.read_csv(csv_file)

        for n in range(10):
            shortname = df.values[n][0].replace('.jpg','')
            rle       = df.values[n][1]
            mask  = run_length_decode(rle,H=CARVANA_HEIGHT, W=CARVANA_WIDTH)
            im_show('mask', mask, resize=0.25)
            cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    check_rle_decode()


    print('\nsucess!')