# find the dice loss limit of resizing
from dataset.carvana_cars import *
from train_seg_net import one_dice_loss_py


def run_find_limit():

    img_dir = CARVANA_DIR + '/annotations/train'  # read all annotations
    H,W = 512, 512  #0.997947997061
    H,W = 256, 256  #0.996000875723
    H,W = 128, 128  #0.99203487
    H,W = 418, 627
    H,W = CARVANA_HEIGHT*2, CARVANA_WIDTH*2


    img_list = glob.glob(img_dir + '/*.gif')
    num_imgs = len(img_list)

    all_loss = 0
    for n in range(num_imgs):
        print('n/num_imgs=%d/%d'%(n,num_imgs))

        img_file = img_list[n]
        img = PIL.Image.open(img_file)
        img = np.array(img)

        label = img

        #downsize
        img = cv2.resize(img.astype(np.float32),(W,H), interpolation=cv2.INTER_LINEAR) #INTER_LINEAR #INTER_CUBIC #INTER_AREA #INTER_LANCZOS4
        img = (img>0.5).astype(np.float32)

        # upsize again
        mask = cv2.resize(img,(CARVANA_WIDTH,CARVANA_HEIGHT), interpolation=cv2.INTER_LINEAR)
        mask = mask >0.5


        #loss
        l = one_dice_loss_py(mask,label)
        all_loss += l

    all_loss = all_loss/num_imgs
    print(all_loss)
    pass


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_train()
    run_find_limit()

    print('\nsucess!')