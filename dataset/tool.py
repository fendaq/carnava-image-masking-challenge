from common import *
# common tool for dataset

# draw -----------------------------------
def im_show(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))

def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)

def draw_text(img, text, pt,  fontScale, color, thickness):

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color, thickness, cv2.LINE_AA)


##http://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=20):

    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if gap==1:
        for p in pts:
            cv2.circle(img,p,thickness,color,-1,cv2.LINE_AA)
    else:
        def pairwise(iterable):
            "s -> (s0, s1), (s2, s3), (s4, s5), ..."
            a = iter(iterable)
            return zip(a, a)

        for p, q in pairwise(pts):
            cv2.line(img,p, q, color,thickness,cv2.LINE_AA)

def draw_dotted_poly(img, pts, color, thickness=1, gap=20):

    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        draw_dotted_line(img,s,e,color,thickness,gap)


def draw_dotted_rect(img, pt1, pt2, color, thickness=1, gap=3):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])]
    draw_dotted_poly(img, pts, color, thickness, gap)


## custom data transform  -----------------------------------

def tensor_to_image(tensor, mean=0, std=1):
    image = tensor.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image*std + mean
    image = image.astype(dtype=np.uint8)
    #img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return image


def tensor_to_label(tensor):
    label = tensor.numpy()*255
    label = label.astype(dtype=np.uint8)
    return label

## transform (input is numpy array, read in by cv2)
def image_to_tensor(image, mean=0, std=1.):
    image = image.astype(np.float32)
    image = (image-mean)/std
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image)   ##.float()
    return tensor

def label_to_tensor(label, threshold=0.5):
    label  = label
    label  = (label>threshold).astype(np.float32)
    tensor = torch.from_numpy(label).type(torch.FloatTensor)
    return tensor

# transform image and label
def randomHorizontalFlip2(image, label, u=0.5):

    if random.random() < u:
        image = cv2.flip(image,1)  #np.fliplr(img)  #cv2.flip(img,1) ##left-right
        label = cv2.flip(label,1)  #np.fliplr(img)  #cv2.flip(img,1) ##left-right

    return image, label


def randomShiftScaleRotate2(image, label, shift_limit=(-0.0625,0.0625), scale_limit=(-0.1,0.1), rotate_limit=(-45,45), aspect_limit = (0,0),  borderMode=cv2.BORDER_CONSTANT, u=0.5):
    #cv2.BORDER_REFLECT_101

    if random.random() < u:
        height,width,channel = image.shape

        angle  = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
        scale  = random.uniform(1+scale_limit[0],1+scale_limit[1])
        aspect = random.uniform(1+aspect_limit[0],1+aspect_limit[1])
        sx    = scale*aspect/(aspect**0.5)
        sy    = scale       /(aspect**0.5)
        dx    = round(random.uniform(shift_limit[0],shift_limit[1])*width )
        dy    = round(random.uniform(shift_limit[0],shift_limit[1])*height)

        cc = math.cos(angle/180*math.pi)*(sx)
        ss = math.sin(angle/180*math.pi)*(sy)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        image = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101


        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width,height])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width+2*dx,height+2*dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        height2,width2  = label.shape
        label = cv2.warpPerspective(label, mat, (width2,height2),flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return image,label


## sampler  -----------------------------------

class FixedSampler(Sampler):
    def __init__(self, data, list):
        self.num_samples = len(list)
        self.list = list

    def __iter__(self):
        #print ('\tcalling Sampler:__iter__')
        return iter(self.list)

    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.num_samples









# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))



