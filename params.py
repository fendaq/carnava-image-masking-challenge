from model.segmentation.SegNet import segnet_vgg
from model.segmentation.my_unet_baseline import UNet1024
from model.segmentation.unet_variant import UNet1024_64in,UNet1024_LKM
from model.segmentation.LKM import LKM
from model.segmentation.FC_DenseNet import my_FCDenseNet

my_computer = True

input_size = 1024

max_epochs = 60

real_batch_size = 15
step_batch_size = 3


#test_batch_size = 8

orig_width = 1918
orig_height = 1280

# threshold = 0.5

#model_factory = segnet_vgg;   save_path = 'test'

#single model train
#model_factory = UNet1024;   save_path = 'unet_double_1024_baseline'
#model_factory = UNet1024_64in; save_path = 'UNet1024_64in'
#model_factory = UNet1024_LKM; save_path = 'uNet1024_LKM'
#model_factory = LKM; save_path = 'LKM'
#model_factory = my_FCDenseNet; save_path = 'my_FCDenseNet'