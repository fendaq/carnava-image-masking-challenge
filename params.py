from model.segmentation.SegNet import segnet_vgg
from model.segmentation.my_unet_baseline import UNet1024, UNet128
from model.segmentation.unet_variant import UNet1024_dropout,\
                        UNet1024_post_train,UNet1024_post_train_01, \
                        UNet1024_GCN_baseline,UNet1024_GCN_k15, \
                        UNet1024_GCN_k15_02,UNet1024_GCN_k15_03, \
                        UNet1024_GCN_k15_04,UNet1024_GCN_k15_05, \
                        UNet1024_GCN_k15_06,UNet1024_GCN_k15_07, \
                        UNet1024_ASPP,UNet1024_ASPP_02,UNet1024_ASPP_03, \
                        UNet1024_ASPP_04,UNet1024_ASPP_05,UNet1024_ASPP_06,\
                        UNet1024_ASPP_07,UNet1024_ASPP_08
from model.segmentation.LKM import LKM,LKM_02
from model.segmentation.FC_DenseNet import my_FCDenseNet,my_FCDenseNet02,FCDenseNet103

my_computer  = True

input_size = 1024

orig_width = 1918
orig_height = 1280

#-----------------------------------------------
# run_valid() run_submit1() save_origin_mask()
npy_BLOCK_SIZE = 8000
model_snap = 'final.pth'

# save_origin_mask()
save_test = True # 'False' means save train images
save_full_resolution_mask = False

#-------carvana_cars.py--------------
post_prosses = False # 'True' return origin sizes, 'False' return input_size

#-------post_train.py----------------
max_post_train_epochs = 50

#post_optimer = 'SGD'
post_optimer = 'Adam'
post_using_ReduceLROnPlateau = True

real_post_batch_size = 15
step_post_batch_size = 3

post_model = UNet1024_post_train_01
#------------------------------------------------

#-------train_seg_net.py-------------

# run_train()
max_epochs = 50

init_checkpoint = None # '060.pth'

#optimer = 'SGD'
optimer = 'Adam'
using_ReduceLROnPlateau = True #bug已修复

LR_sgd = [ (0, 0.01),  (35, 0.005),  (40,0.001),  (45, 0.0002),(55, -1)]
LR_adam = [ (0, 0.001),  (35, 0.0005),  (55, -1)]

real_batch_size = 15
step_batch_size = 3

# test_batch_size = 8

# threshold = 0.5

#model_factory = UNet1024_GCN_k15_04;   save_path = 'test'

#single model train
#model_factory = UNet1024;   save_path = 'unet_double_1024_baseline_adam'
#model_factory = UNet1024;   save_path = 'unet_double_1024_baseline_1280'
#model_factory = UNet1024;   save_path = 'unet_double_1024_baseline_flip'
#model_factory = UNet1024_dropout; save_path = 'UNet1024_dropout'
#model_factory = UNet1024; save_path = 'unet1024_batch_2'

#model_factory = UNet1024_ASPP; save_path = 'unet1024_ASPP'
#model_factory = UNet1024_ASPP_02; save_path = 'UNet1024_ASPP_02'
#model_factory = UNet1024_ASPP_03;   save_path = 'UNet1024_ASPP_03'
#model_factory = UNet1024_ASPP_04; save_path = 'UNet1024_ASPP_04'
#model_factory = UNet1024_ASPP_07; save_path = 'UNet1024_ASPP_07'
model_factory = UNet1024_ASPP_08; save_path = 'UNet1024_ASPP_08'

#model_factory = UNet1024_GCN; save_path = 'UNet1024_GCN'
#model_factory = UNet1024_GCN_k15_04;   save_path = 'UNet1024_GCN_k15_04'
#model_factory = UNet1024_GCN_k15_05; save_path = 'UNet1024_GCN_k15_05'
#model_factory = UNet1024_GCN_k15_06; save_path = 'UNet1024_GCN_k15_06'
#model_factory = UNet1024_GCN_k15_07; save_path = 'UNet1024_GCN_k15_07'

#model_factory = LKM; save_path = 'LKM_152'
#model_factory = LKM; save_path = 'LKM_152_k15'
#model_factory = LKM_02; save_path = 'LKM_02'

#model_factory = my_FCDenseNet; save_path = 'my_FCDenseNet'

#model_factory = segnet_vgg;   save_path = 'segnet_vgg'