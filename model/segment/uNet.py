# uNet from scratch

from common import *

import torch
import torch.nn as nn
import torch.nn.functional as F

## -------------------------------------------------------------------------------------
# note! bias = flase,inplace = true
def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]

class UNet_double_1024_5 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet_double_1024_5, self).__init__()
        in_channels, height, width = in_shape


        #512
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        #256


        #UNet512_2 ------------------------------------------------------------------------
        #256
        self.down2 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down5 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down6 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up6 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up5 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up4 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up3 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up2 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
        )
        #128
        #-------------------------------------------------------------------------

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu(  3+16,8, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     8,8, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     8,8, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(8, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #512
        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #64

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #64

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) #32

        down5 = self.down5(out)
        out   = F.max_pool2d(down5, kernel_size=2, stride=2) #16

        down6 = self.down6(out)
        out   = F.max_pool2d(down6, kernel_size=2, stride=2) # 8

        out   = self.center(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down6, out],1)
        out   = self.up6(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down5, out],1)
        out   = self.up5(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #512
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        #out   = F.upsample_bilinear(out, scale_factor=2) #1024
        #x     = F.upsample_bilinear(x,   scale_factor=2)
        out   = torch.cat([x, out],1)
        out   = self.up0(out)


        out   = self.classify(out)

        return out















# https://gist.github.com/fsodogandji/e69dfecf153d4df62044b8ca385c4577 ----------------------------------------------------