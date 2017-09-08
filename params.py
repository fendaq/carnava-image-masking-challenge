from model.segmentation.SegNet import segnet_vgg
from model.segmentation.my_unet_baseline import UNet1024, UNet128
from model.segmentation.unet_variant import UNet1024_01,UNet1024_GCN,UNet1024_GCN_baseline,UNet1024_GCN_k15,UNet1024_ASPP
from model.segmentation.LKM import LKM,LKM_02
from model.segmentation.FC_DenseNet import my_FCDenseNet02,my_FCDenseNet,FCDenseNet103

my_computer  = True

input_size = 1024

#----保存完整mask并后处理----------------
#-------用于save_origin_mask()---------
save_test = False
save_full_resolution_mask = False
#-------carvana_cars.py---------------
post_prosses = False #true则迭代返回原图尺寸,false则为input_size
#-------------------------

max_epochs = 42

optimer = 'SGD'
#optimer = 'Adam'
using_ReduceLROnPlateau = True #bug已修复

real_batch_size = 15
step_batch_size = 3


#test_batch_size = 8

orig_width = 1918
orig_height = 1280

# threshold = 0.5

#model_factory = UNet128;   save_path = 'test'

#single model train
model_factory = UNet1024;   save_path = 'unet_double_1024_baseline'
#model_factory = UNet1024;   save_path = 'unet_double_1024_baseline_1280'

#model_factory = UNet1024_ASPP; save_path = 'unet1024_ASPP'

#model_factory = UNet1024_GCN; save_path = 'UNet1024_GCN'

#model_factory = LKM; save_path = 'LKM_152'
#model_factory = LKM; save_path = 'LKM_152_k15'
#model_factory = LKM_02; save_path = 'LKM_02'

#model_factory = my_FCDenseNet; save_path = 'my_FCDenseNet'

#model_factory = segnet_vgg;   save_path = 'segnet_vgg'