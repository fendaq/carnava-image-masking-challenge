#large kernel matters
#https://github.com/ycszen/pytorch-ss/blob/master/gcn.py note errors in that file
from common import *
from model.segmentation.blocks import *

import torch.nn as nn
import torchvision.models as models

class LKM(nn.Module):
    def __init__(self, in_shape, num_classes=1):
        super(LKM, self).__init__()

        self.num_classes = num_classes

        #resnet = models.resnet50(pretrained=True)
        resnet = models.ResNet(models.resnet.Bottleneck, [3, 4, 6, 3])
        resnet.load_state_dict(torch.load('/home/lhc/.torch/models/resnet50-19c8e357.pth'))


        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcn1 = GCN(2048, self.num_classes)
        self.gcn2 = GCN(1024, self.num_classes)
        self.gcn3 = GCN(512, self.num_classes)
        self.gcn4 = GCN(256, self.num_classes)
        #self.gcn5 = GCN(64, self.num_classes)

        self.refine1 = Refine(self.num_classes)
        self.refine2 = Refine(self.num_classes)
        self.refine3 = Refine(self.num_classes)
        self.refine4 = Refine(self.num_classes)
        self.refine5 = Refine(self.num_classes)
        self.refine6 = Refine(self.num_classes)
        self.refine7 = Refine(self.num_classes)
        self.refine8 = Refine(self.num_classes)
        self.refine9 = Refine(self.num_classes)
        self.refine10 = Refine(self.num_classes)

        self.dconv5 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)
        self.dconv4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)
        self.dconv3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)
        self.dconv2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)
        self.dconv1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)

        '''
        self.out0 = self._classifier(2048)
        self.out1 = self._classifier(1024)
        self.out2 = self._classifier(512)
        self.out_e = self._classifier(256)
        self.out3 = self._classifier(64)
        self.out4 = self._classifier(64)
        self.out5 = self._classifier(32)

        self.transformer = nn.Conv2d(256, 64, kernel_size=1)

    def _classifier(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes/2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes/2, self.num_classes, 1),
        )
        '''

    def forward(self, x, is_dconv=True, is_test=False):
        input = x #1024
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x #512
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)   #256
        fm2 = self.layer2(fm1) #128
        fm3 = self.layer3(fm2) #64
        fm4 = self.layer4(fm3) #32

        gcfm4 = self.refine1(self.gcn1(fm4)) #32
        gcfm3 = self.refine2(self.gcn2(fm3)) #64
        gcfm2 = self.refine3(self.gcn3(fm2)) #128
        gcfm1 = self.refine4(self.gcn4(fm1)) #256
        #gcfm5 = self.refine5(self.gcn5(conv_x))

        if is_dconv == False:
            fs4 = self.refine6(F.upsample_bilinear(gcfm4, fm3.size()[2:]) + gcfm3) #64
            fs3 = self.refine7(F.upsample_bilinear(fs4, fm2.size()[2:]) + gcfm2)   #128
            fs2 = self.refine8(F.upsample_bilinear(fs3, fm1.size()[2:]) + gcfm1)   #256
            fs1 = self.refine9(F.upsample_bilinear(fs2, conv_x.size()[2:]))        #512
            out = self.refine10(F.upsample_bilinear(fs1, input.size()[2:]))
            if is_test:
                print(fm1.size(),fm2.size(),fm3.size(),fm4.size())
                print(fs1.size(),fs2.size(),fs3.size(),fs4.size())
        else:
            #Dconv
            fs4 = self.refine6(self.dconv5(gcfm4) + gcfm3) #64
            fs3 = self.refine7(self.dconv4(fs4) + gcfm2)   #128
            fs2 = self.refine8(self.dconv3(fs3) + gcfm1)   #256
            fs1 = self.refine9(self.dconv2(fs2))      #512
            out = self.refine10(self.dconv1(fs1))
            if is_test:
                print(fm1.size(),fm2.size(),fm3.size(),fm4.size())
                print(fs1.size(),fs2.size(),fs3.size(),fs4.size())

        out = torch.squeeze(out, dim=1)

        return out #, fs4, fs3, fs2, fs1, gcfm1

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    CARVANA_HEIGHT = 1280
    CARVANA_WIDTH  = 1918
    batch_size  = 2
    C,H,W = 3,512,512    #3,CARVANA_HEIGHT,CARVANA_WIDTH

    if 1: # BCELoss2d()
        num_classes = 1

        inputs = torch.randn(batch_size,C,H,W)
        labels = torch.LongTensor(batch_size,H,W).random_(1).type(torch.FloatTensor)

        net = LKM(in_shape=(C,H,W)).cuda().train()
        x = Variable(inputs.cuda())
        y = Variable(labels.cuda())
        logits = net.forward(x,is_dconv=True,is_test=True)
        print(logits.size())

        loss = BCELoss2d()(logits, y)
        loss.backward()

        print(type(net))
        #print(net)
        print('logits')
        print(logits)
    #input('Press ENTER to continue.')
