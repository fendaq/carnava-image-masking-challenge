from model.segmentation.SegNet import segnet_vgg
from model.segmentation.LKM import LKM,LKM_02
from model.segmentation.FC_DenseNet import my_FCDenseNet,my_FCDenseNet02,FCDenseNet103
from model.segmentation.my_unet_baseline import UNet1024, UNet128
from model.segmentation.unet_variant import \
            UNet1024_post_train,UNet1024_post_train_01, \
            UNet1024_post_train_02, \
            UNet1024_GCN_k15_06, \
            UNet1024_ASPP_02,UNet1024_ASPP_07,UNet1024_ASPP_08
'''
            UNet1024_dropout,
            UNet1024_ASPP,UNet1024_ASPP_03,UNet1024_ASPP_04,UNet1024_ASPP_05,
            UNet1024_ASPP_06,
            UNet1024_GCN_baseline,UNet1024_GCN_k15,
            UNet1024_GCN_k15_02,UNet1024_GCN_k15_03,
            UNet1024_GCN_k15_04,UNet1024_GCN_k15_05,
            UNet1024_GCN_k15_07,
'''         

#my_computer  = True
#data_dir-----------------------
CARVANA_DIR = '/Kaggle/kaggle-carvana-cars-2017'
#CARVANA_DIR = '/kaggle_data_results/Kaggle/kaggle-carvana-cars-2017'
#-------------------------------

#out_dir------------------------
out_dir = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/ensemble/'
#out_dir = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/'

#out_dir = '/kaggle_data_results/results/lhc/ensemble/'
#out_dir = '/kaggle_data_results/results/lhc/single/'
#-------------------------------

#input_size = 1024
input_w = 1024
input_h = 1024

orig_width = 1918
orig_height = 1280

#-----------------------------------------------
# train_seg: run_valid() run_submit1() save_origin_mask()
npy_BLOCK_SIZE = 8000 #both for train_seg and post_train
model_snap = None #'060.pth'

#-------carvana_cars.py--------------
post_prosses = False # 'True' return origin sizes, 'False' return input_size

#-------post_train.py----------------
# k-folds
post_k_folds = 1

#run_post_train
max_post_train_epochs = 65
init_post = '050.pth'

#post_optimer = 'SGD'
post_optimer = 'Adam'
post_using_ReduceLROnPlateau = True

#real_post_batch_size = 15
#step_post_batch_size = 3
post_submit_snap = 'final.pth'

post_model = UNet1024_post_train_02
#------------------------------------------------

#-------train_seg_net.py-------------
# k-folds
k_version = 2
k_folds = 1

#ensemble_train = True

# run_train()
max_epochs = 60

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

#model_factory = UNet1024_ASPP_08;   save_path = 'test'

#single model train
#model_factory = UNet1024;   save_path = 'unet_double_1024_baseline_adam'
#model_factory = UNet1024;   save_path = 'unet_double_1024_baseline_1280'
#model_factory = UNet1024;   save_path = 'unet_double_1024_baseline_flip'

#model_factory = UNet1024_ASPP_02; save_path = 'UNet1024_ASPP_02'
#model_factory = UNet1024_ASPP_03;   save_path = 'UNet1024_ASPP_03'

#model_factory = UNet1024_ASPP_07; save_path = 'UNet1024_ASPP_07'
model_factory = UNet1024_ASPP_08; save_path = 'UNet1024_ASPP_08_k1'
#model_factory = UNet1024_ASPP_08; save_path = 'UNet1024_ASPP_08_k2'
#model_factory = UNet1024_ASPP_08; save_path = 'UNet1024_ASPP_08_k3'
#model_factory = UNet1024_ASPP_08; save_path = 'UNet1024_ASPP_08_k4'
#model_factory = UNet1024_ASPP_08; save_path = 'UNet1024_ASPP_08_k5'

#model_factory = UNet1024_GCN_k15_06; save_path = 'UNet1024_GCN_k15_06'
