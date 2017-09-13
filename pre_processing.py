#亮度矫正

import os

import numpy as np
import cv2

#https://github.com/Hsuxu/carvana-pytorch-uNet/blob/master/preprocessing.py
def contrast_adjust(image, alpha=1.3, beta=20):
    """
    adjust constrast through gamma correction
    newimg = image * alpha + beta
    input:
        image: np.uint8 or np.float32
    output:
        image: np.uint8 or np.float
    """
    newimage = image.astype(np.float32) * alpha + beta
    
    if type(image[0,0,0])==np.uint8:
        newimage[newimage < 0] = 0
        newimage[newimage > 255] = 255
        return np.uint8(newimage)
    else:
        newimage[newimage < 0] = 0
        newimage[newimage > 1] = 1.
        return newimage

#https://zhuanlan.zhihu.com/p/24425116
def gamma_preprocess():

    img = cv2.imread('/Kaggle/test_pre_process/0d1a9caf4350_11.jpg')

    # 分通道计算每个通道的直方图
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

    # 定义Gamma矫正的函数
    def gamma_trans(img, gamma):
        # 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
        gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        
        # 实现这个映射用的是OpenCV的查表函数
        return cv2.LUT(img, gamma_table)

    # 执行Gamma矫正，小于1的值让暗部细节大量提升，同时亮部细节少量提升
    img_corrected = gamma_trans(img, 0.9)
    cv2.imwrite('/Kaggle/test_pre_process/img_corrected.jpg', img_corrected)

    # 分通道计算Gamma矫正后的直方图
    hist_b_corrected = cv2.calcHist([img_corrected], [0], None, [256], [0, 256])
    hist_g_corrected = cv2.calcHist([img_corrected], [1], None, [256], [0, 256])
    hist_r_corrected = cv2.calcHist([img_corrected], [2], None, [256], [0, 256])

    # 将直方图进行可视化
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()

    pix_hists = [
        [hist_b, hist_g, hist_r],
        [hist_b_corrected, hist_g_corrected, hist_r_corrected]
    ]

    pix_vals = range(256)
    for sub_plt, pix_hist in zip([121, 122], pix_hists):
        ax = fig.add_subplot(sub_plt, projection='3d')
        for c, z, channel_hist in zip(['b', 'g', 'r'], [20, 10, 0], pix_hist):
            cs = [c] * 256
            ax.bar(pix_vals, channel_hist, zs=z, zdir='y', color=cs, alpha=0.618, edgecolor='none', lw=0)

        ax.set_xlabel('Pixel Values')
        ax.set_xlim([0, 256])
        ax.set_ylabel('Channels')
        ax.set_zlabel('Counts')

    plt.show()

#https://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
def histequal_preprocess():

    def hisEqulColor(img):
        ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
        channels=cv2.split(ycrcb)
        cv2.equalizeHist(channels[0],channels[0])
        cv2.merge(channels,ycrcb)
        cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
        return img

    img = cv2.imread('/Kaggle/test_pre_process/0d1a9caf4350_11.jpg')
    equ = hisEqulColor(img)
    cv2.imwrite('/Kaggle/test_pre_process/equ.jpg', equ)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    gamma_preprocess()
    #histequal_preprocess()

    print('\nsucess!')