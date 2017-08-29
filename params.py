from model.segmentation.SegNet import segnet_vgg
from model.segmentation.my_unet_baseline import UNet1024

input_size = 1024

max_epochs = 50

real_batch_size = 32
step_batch_size = 4


#test_batch_size = 8

orig_width = 1918
orig_height = 1280

# threshold = 0.5


model_factory = UNet1024;   save_path = 'UNet1024-baseline'