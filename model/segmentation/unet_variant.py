# unet from scratch
from common import *
from model.segmentation.loss import *
from model.segmentation.blocks import *

import torch
import torch.nn as nn
import torch.nn.functional as F

# baseline 128x128, 256x256, 512x512, 1024x1024 for experiments -----------------------------------------------

BN_EPS = 1e-4  #1e-4  #1e-5


# 1024x1024
class UNet1024_64in (nn.Module):
    def __init__(self, in_shape):
        super(UNet1024, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #1024
        self.down1 = StackEncoder(  C,   64, kernel_size=3)   #512
        self.down2 = StackEncoder( 64,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 32
        self.down6 = StackEncoder(512,  1024, kernel_size=3)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  64, kernel_size=3)  #256
        self.up1 = StackDecoder(  64,  64,  64, kernel_size=3)  #512
        self.classify = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ;print('down1',down1.size())  #256
        down2,out = self.down2(out)   ;print('down2',down2.size())  #128
        down3,out = self.down3(out)   ;print('down3',down3.size())  #64
        down4,out = self.down4(out)   ;print('down4',down4.size())  #32
        down5,out = self.down5(out)   ;print('down5',down5.size())  #16
        down6,out = self.down6(out)   ;print('down6',down6.size())  #8
        pass                          ;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out


class UNet1024_LKM (nn.Module): #Large kernel matters
        def __init__(self, in_shape):
        super(UNet1024, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #1024
        self.down1 = StackEncoder(  C,   64, kernel_size=3)   #512
        self.down2 = StackEncoder( 64,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 32
        self.down6 = StackEncoder(512,  1024, kernel_size=3)   # 16
        
        '''
        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )
        '''
        self.center = 


        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  64, kernel_size=3)  #256
        self.up1 = StackDecoder(  64,  64,  64, kernel_size=3)  #512
        self.classify = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ;print('down1',down1.size())  #256
        down2,out = self.down2(out)   ;print('down2',down2.size())  #128
        down3,out = self.down3(out)   ;print('down3',down3.size())  #64
        down4,out = self.down4(out)   ;print('down4',down4.size())  #32
        down5,out = self.down5(out)   ;print('down5',down5.size())  #16
        down6,out = self.down6(out)   ;print('down6',down6.size())  #8
        pass                          ;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out

class UNet1024_ASPP (nn.Module):
        def __init__(self, in_shape):
        super(UNet1024, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #1024
        self.down1 = StackEncoder(  C,   64, kernel_size=3)   #512
        self.down2 = StackEncoder( 64,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 32
        self.down6 = StackEncoder(512,  1024, kernel_size=3)   # 16
        
        '''
        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )
        '''
        self.center = 

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  64, kernel_size=3)  #256
        self.up1 = StackDecoder(  64,  64,  64, kernel_size=3)  #512
        self.classify = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ;print('down1',down1.size())  #256
        down2,out = self.down2(out)   ;print('down2',down2.size())  #128
        down3,out = self.down3(out)   ;print('down3',down3.size())  #64
        down4,out = self.down4(out)   ;print('down4',down4.size())  #32
        down5,out = self.down5(out)   ;print('down5',down5.size())  #16
        down6,out = self.down6(out)   ;print('down6',down6.size())  #8
        pass                          ;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    CARVANA_HEIGHT = 1280
    CARVANA_WIDTH  = 1918
    batch_size  = 1
    C,H,W = 3,256,256    #3,CARVANA_HEIGHT,CARVANA_WIDTH

    if 1: # BCELoss2d()
        num_classes = 1

        inputs = torch.randn(batch_size,C,H,W)
        labels = torch.LongTensor(batch_size,H,W).random_(1).type(torch.FloatTensor)

        net = UNet1024_64in(in_shape=(C,H,W)).cuda().train()
        x = Variable(inputs.cuda())
        y = Variable(labels.cuda())
        logits = net.forward(x)

        loss = BCELoss2d()(logits, y)
        loss.backward()

        print(type(net))
        print(net)
        print('logits')
        print(logits)
    #input('Press ENTER to continue.')


