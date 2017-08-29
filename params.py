from model.segment.uNet import UNet_double_1024, UNet_double_1024_deconv
# from model.segment.DeepLabv2 import Deeplab_v2_resnet
from model.segment.DeepLabv3 import Deeplabv3_parallel
from model.segment.SegNet import segnet_vgg

input_size = 1024

max_epochs = 50

batch_size = 4
test_batch_size = 8

orig_width = 1918
orig_height = 1280

# threshold = 0.5

model_factory = UNet_double_1024