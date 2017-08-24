from common import *
from dataset.mask import *


CARVANA_DIR       = '/root/share/data/kaggle-carvana-cars-2017'
CARVANA_NUM_VIEWS = 16
CARVANA_HEIGHT = 1280
CARVANA_WIDTH  = 1918
CARVANA_H = 512
CARVANA_W = 512



#data iterator ----------------------------------------------------------------

class KgCarDataset_1x(Dataset):

    def __init__(self, split, transform=[], is_label=True, is_preload=True):
        channel,height,width = 3, CARVANA_H, CARVANA_W

        # read names
        split_file = CARVANA_DIR +'/split/'+ split
        with open(split_file) as f:
            names = f.readlines()
        names = [name.strip()for name in names]
        num   = len(names)

        #read images
        images = None
        if is_preload==True:
            images = np.zeros((num,height,width,channel),dtype=np.float32)
            for n in range(num):
                name = names[n]
                img_file = CARVANA_DIR + '/images/%s.jpg'%(name)
                img = cv2.imread(img_file)

                #img = cv2.cvtColor(img,cv2.COLOR_fBGR2HSV)
                #img = cv2.resize(img,(width,height))
                images[n] = img/255.

                #debug
                #print(n)

        #read labels
        labels = None
        if is_label==True:
            labels = np.zeros((num,height,width),dtype=np.float32)
            for n in range(num):
                name = names[n]
                shortname = name.split('/')[-1]
                # mask_file = CARVANA_DIR + '/annotations/%s_mask.gif'%(name)
                # mask = PIL.Image.open(mask_file)   #opencv does not read gif
                # mask = np.array(mask)

                mask_file = CARVANA_DIR + '/annotations/%s_mask.png'%(name)
                mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
                #mask = cv2.resize(mask,(width,height))
                labels[n] = mask/255

                #debug
                if 0:
                    im_show('mask1', mask*255, resize=1)
                    cv2.waitKey(0)


        #save
        self.split     = split
        self.transform = transform
        self.names  = names
        self.images = images
        self.labels = labels



    #https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def __getitem__(self, index):

        if self.images is None:
            name = self.names[index]
            img_file = CARVANA_DIR + '/images/%s.jpg'%(name)
            img   = cv2.imread(img_file)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

            image = img.astype(np.float32)/255
        else:
            image = self.images[index]


        if self.labels is None:
            for t in self.transform:
                image = t(image)
            image = image_to_tensor(image)
            return image, index

        else:
            label = self.labels[index]
            for t in self.transform:
                image,label = t(image,label)
            image = image_to_tensor(image)
            label = label_to_tensor(label)
            return image, label, index


    def __len__(self):
        #print ('\tcalling Dataset:__len__')
        return len(self.names)


# 2x label
class KgCarDataset(Dataset):

    def __init__(self, split, transform=[], is_label=True, is_preload=True):
        channel,height,width = 3, CARVANA_H, CARVANA_W

        # read names
        split_file = CARVANA_DIR +'/split/'+ split
        with open(split_file) as f:
            names = f.readlines()
        names = [name.strip()for name in names]
        num   = len(names)

        #read images
        images = None
        if is_preload==True:
            images = np.zeros((num,height,width,channel),dtype=np.float32)
            for n in range(num):
                name = names[n]
                img_file = CARVANA_DIR + '/images/%s.jpg'%(name)
                img = cv2.imread(img_file)

                #img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                #img = cv2.resize(img,(width,height))
                images[n] = img/255.

                #debug
                #print(n)

        #read labels
        labels = None
        if is_label==True:
            labels = np.zeros((num,2*height,2*width),dtype=np.float32)
            for n in range(num):
                name = names[n]
                name = name.replace('%dx%d'%(height,width),'%dx%d'%(2*height,2*width))

                mask_file = CARVANA_DIR + '/annotations/%s_mask.png'%(name)
                mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
                #mask = cv2.resize(mask,(width,height))
                labels[n] = mask/255

                #debug
                if 0:
                    im_show('mask1', mask*255, resize=1)
                    cv2.waitKey(0)


        #save
        self.split      = split
        self.transform  = transform
        self.names  = names
        self.images = images
        self.labels = labels



    #https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def __getitem__(self, index):

        if self.images is None:
            name = self.names[index]
            img_file = CARVANA_DIR + '/images/%s.jpg'%(name)
            img   = cv2.imread(img_file)
            image = img.astype(np.float32)/255
        else:
            image = self.images[index]


        if self.labels is None:
            for t in self.transform:
                image = t(image)
            image = image_to_tensor(image)
            return image, index

        else:
            label = self.labels[index]
            for t in self.transform:
                image,label = t(image,label)
            image = image_to_tensor(image)
            label = label_to_tensor(label)
            return image, label, index

    def __len__(self):
        #print ('\tcalling Dataset:__len__')
        return len(self.names)

#-----------------------------------------------------------------------

def check_dataset(dataset, loader):

    if dataset.labels is not None:
        for i, (images, labels, indices) in enumerate(loader, 0):
            print('i=%d: '%(i))

            num = len(images)
            for n in range(num):
                image = images[n]
                label = labels[n]
                image = tensor_to_image(image, std=255)
                label = tensor_to_label(label)

                im_show('image', image, resize=1)
                im_show('label', label, resize=1)
                cv2.waitKey(1)





def run_check_dataset():
    dataset = KgCarDataset( 'train%dx%d_5088'%(CARVANA_H,CARVANA_W),  #'train_5088'
                                transform=[
                                    lambda x,y:  randomShiftScaleRotate2(x,y,shift_limit=(-0.0625,0.0625), scale_limit=(-0.0,0.0),  aspect_limit = (1-1/1.2   ,1.2-1), rotate_limit=(0,0)),
                                    lambda x,y:  randomHorizontalFlip2(x,y),
                                ],
                            is_preload=False,
                         )

    if 1: #check indexing
        for n in range(100):
            image, label, index = dataset[n]
            image = tensor_to_image(image, std=255)
            label = tensor_to_label(label)

            im_show('image', image, resize=1)
            im_show('label', label, resize=1)
            cv2.waitKey(0)

    if 0: #check iterator
        #sampler = FixedSampler(dataset, ([4]*100))
        sampler = SequentialSampler(dataset)
        loader  = DataLoader(dataset, batch_size=4, sampler=sampler,  drop_last=False, pin_memory=True)

        for epoch in range(100):
            print('epoch=%d -------------------------'%(epoch))
            check_dataset(dataset, loader)




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_dataset()


    print('\nsucess!')