#origin https://github.com/SimJeg/FC-DenseNet/blob/master/FC-DenseNet.py
#https://github.com/bfortuner/pytorch_tiramisu/blob/master/tiramisu-pytorch.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

def center_crop(layer, max_height, max_width):
    #https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/merge.py#L162
    #Author does a center crop which crops both inputs (skip and upsample) to size of minimum dimension on both w/h
    batch_size, n_channels, layer_height, layer_width = layer.size()
    xy1 = (layer_width - max_width) // 2
    xy2 = (layer_height - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, drop_rate=0.2):
        super(_DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu', nn.ReLU(inplace=True)),
        self.add_module('conv', nn.Conv2d(num_input_features, growth_rate, kernel_size=3, stride=1, bias=True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, upsample=False):
        super(_DenseBlock, self).__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([_DenseLayer(num_input_features 
                                      + i*growth_rate,growth_rate) for i in range(num_layers)])
    
    def forward(self, x):
        if self.upsample:
            pass
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x,out], 1)
            return x



class _TransitionDown(nn.Sequential):
    def __init__(self, num_input_features, drop_rate=0.2):
        super(_TransitionDown, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_input_features,
                                          kernel_size=1, stride=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(kernel_size=2, stride=2))
    
    def forward(self, x):
        return super(_TransitionDown, self).forward(x)

class _TransitionUp(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionUp, self).__init__()
        self.add_module('conv', nn.ConvTranspose2d(num_input_features, num_output_features,
                                          kernel_size=3, stride=2, bias=True))

class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(Bottleneck, self).__init__()
        self.add_module('bottleneck', _DenseBlock(in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super(Bottleneck, self).forward(x)


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5), 
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5, 
                 growth_rate=16, out_chans_first_conv=48, n_classes=1):
        super(FCDenseNet, self).__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks

        cur_channels_count = 0
        skip_connection_channel_counts = []

        #####################
        # First Convolution #
        #####################

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels, 
                  out_channels=out_chans_first_conv, kernel_size=3, 
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################
        
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                _DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(_TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################
        
        self.add_module('bottleneck',Bottleneck(cur_channels_count, 
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels 

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(_TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(_DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i], 
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        #One final dense block
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1], 
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        #####################
        #      Softmax      #
        #####################

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count, 
               out_channels=n_classes, kernel_size=1, stride=1, 
                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax()

    def forward(self, x, is_test=False):
        
        if is_test:    
            print("INPUT",x.size())
        
        out = self.firstconv(x)
        
        skip_connections = []
        for i in range(len(self.down_blocks)):

            if is_test:
                print("DBD size",out.size())
            
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
            
        out = self.bottleneck(out)
        
        if is_test:
            print ("bnecksize",out.size())
        
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop() 
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)
            if is_test:
                print("DOWN_SKIP_PRE_UPSAMPLE",out.size(),skip.size())
                print("DOWN_SKIP_AFT_UPSAMPLE",out.size(),skip.size())
            
        out = self.finalConv(out)
        out = self.softmax(out)

        
        return out

def FCDenseNet57(n_classes):
    return FCDenseNet(in_channels=3, down_blocks=(4, 4, 4, 4, 4), 
                 up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4, 
                 growth_rate=12, out_chans_first_conv=48, n_classes=n_classes)

def FCDenseNet67(n_classes):
    return FCDenseNet(in_channels=3, down_blocks=(5, 5, 5, 5, 5), 
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5, 
                 growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)

def FCDenseNet103(in_shape, n_classes=1):
    return FCDenseNet(in_channels=3, down_blocks=(4,5,7,10,12), 
                 up_blocks=(12,10,7,5,4), bottleneck_layers=15, 
                 growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    CARVANA_HEIGHT = 1280
    CARVANA_WIDTH  = 1918
    batch_size  = 2
    C,H,W = 3,1024,1024    #3,CARVANA_HEIGHT,CARVANA_WIDTH

    if 1: # BCELoss2d()
        num_classes = 1

        inputs = torch.randn(batch_size,C,H,W)
        labels = torch.LongTensor(batch_size,H,W).random_(1).type(torch.FloatTensor)

        net = FCDenseNet103(in_shape=(C,H,W)).cuda().train()
        x = Variable(inputs.cuda())
        y = Variable(labels.cuda())
        logits = net.forward(x,is_test=True)
        print(logits.size())

        loss = BCELoss2d()(logits, y)
        loss.backward()

        print(type(net))
        #print(net)
        print('logits')
        print(logits)
    #input('Press ENTER to continue.')
