#https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet_efficient.py#L16

# This implementation is a new efficient implementation of Densenet-BC,
# as described in "Memory-Efficient Implementation of DenseNets"

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul
from collections import OrderedDict
from torch.autograd import Variable, Function
from torch._thnn import type2backend
from torch.backends import cudnn

from model.segmentation.loss import *


# I'm throwing all the gross code at the end of the file :)
# Let's start with the nice (and interesting) stuff

def center_crop(layer, max_height, max_width):
    #https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/merge.py#L162
    #Author does a center crop which crops both inputs (skip and upsample) to size of minimum dimension on both w/h
    batch_size, n_channels, layer_height, layer_width = layer.size()
    xy1 = (layer_width - max_width) // 2
    xy2 = (layer_height - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

class _SharedAllocation(object):
    """
    A helper class which maintains a shared memory allocation.
    Used for concatenation and batch normalization.
    """
    def __init__(self, storage):
        self.storage = storage

    def type(self, t):
        self.storage = self.storage.type(t)

    def type_as(self, obj):
        if isinstance(obj, Variable):
            self.storage = self.storage.type(obj.data.storage().type())
        elif isinstance(obj, torch._TensorBase):
            self.storage = self.storage.type(obj.storage().type())
        else:
            self.storage = self.storage.type(obj.type())

    def resize_(self, size):
        if self.storage.size() < size:
            self.storage.resize_(size)
        return self


class _EfficientDensenetBottleneck(nn.Module):
    """
    A optimized layer which encapsulates the batch normalization, ReLU, and
    convolution operations within the bottleneck of a DenseNet layer.

    This layer usage shared memory allocations to store the outputs of the
    concatenation and batch normalization features. Because the shared memory
    is not perminant, these features are recomputed during the backward pass.
    """
    def __init__(self, shared_allocation_1, shared_allocation_2, num_input_channels, num_output_channels):

        super(_EfficientDensenetBottleneck, self).__init__()
        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2
        self.num_input_channels = num_input_channels

        self.norm_weight = nn.Parameter(torch.Tensor(num_input_channels))
        self.norm_bias = nn.Parameter(torch.Tensor(num_input_channels))
        self.register_buffer('norm_running_mean', torch.zeros(num_input_channels))
        self.register_buffer('norm_running_var', torch.ones(num_input_channels))
        self.conv_weight = nn.Parameter(torch.Tensor(num_output_channels, num_input_channels, 1, 1))
        self._reset_parameters()


    def _reset_parameters(self):
        self.norm_running_mean.zero_()
        self.norm_running_var.fill_(1)
        self.norm_weight.data.uniform_()
        self.norm_bias.data.zero_()
        stdv = 1. / math.sqrt(self.num_input_channels)
        self.conv_weight.data.uniform_(-stdv, stdv)


    def forward(self, inputs):
        if isinstance(inputs, Variable):
            inputs = [inputs]
        fn = _EfficientDensenetBottleneckFn(self.shared_allocation_1, self.shared_allocation_2,
            self.norm_running_mean, self.norm_running_var,
            stride=1, padding=0, dilation=1, groups=1,
            training=self.training, momentum=0.1, eps=1e-5)
        return fn(self.norm_weight, self.norm_bias, self.conv_weight, *inputs)


class _DenseLayer(nn.Sequential):
    def __init__(self, shared_allocation_1, shared_allocation_2, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2
        self.drop_rate = drop_rate

        self.add_module('bn', _EfficientDensenetBottleneck(shared_allocation_1, shared_allocation_2,
            num_input_features, bn_size * growth_rate))
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),

    def forward(self, x):
        if isinstance(x, Variable):
            prev_features = [x]
        else:
            prev_features = x
        new_features = super(_DenseLayer, self).forward(prev_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class _TransitionDown(nn.Sequential):
    def __init__(self, num_input_features):
        super(_TransitionDown, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_input_features,
                                          kernel_size=1, stride=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(kernel_size=2, stride=2))
    
    def forward(self, x):
        return super(_TransitionDown, self).forward(x)

class _TransitionUp(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionUp, self).__init__()
        self.convTrans = nn.ConvTranspose2d(num_input_features, num_output_features,
                                          kernel_size=3, stride=2, bias=True)
    
    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out

class _DenseBlock(nn.Container):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, storage_size=1024):
        input_storage_1 = torch.Storage(storage_size)
        input_storage_2 = torch.Storage(storage_size)
        self.final_num_features = num_input_features + (growth_rate * num_layers)
        self.shared_allocation_1 = _SharedAllocation(input_storage_1)
        self.shared_allocation_2 = _SharedAllocation(input_storage_2)

        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(self.shared_allocation_1, self.shared_allocation_2, num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


    def forward(self, x):
        # Update storage type
        self.shared_allocation_1.type_as(x)
        self.shared_allocation_2.type_as(x)

        # Resize storage
        final_size = list(x.size())
        final_size[1] = self.final_num_features
        final_storage_size = reduce(mul, final_size, 1)
        self.shared_allocation_1.resize_(final_storage_size)
        self.shared_allocation_2.resize_(final_storage_size)

        outputs = [x]
        for module in self.children():
            outputs.append(module.forward(outputs))
        return torch.cat(outputs, dim=1)


class DenseNetEfficient(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    This model uses shared memory allocations for the outputs of batch norm and
    concat operations, as described in `"Memory-Efficient Implementation of DenseNets"`.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0, avgpool_size=8,
                 num_classes=10):

        super(DenseNetEfficient, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = avgpool_size

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features
                                                            * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(
            features.size(0), -1)
        out = self.classifier(out)
        return out

class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5), up_blocks=(5,5,5,5,5),
                 bottleneck_layers=5, bn_size=4, drop_rate=0.2,
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
                _DenseBlock(down_blocks[i], cur_channels_count, bn_size, growth_rate, drop_rate))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(_TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################
        
        self.add_module('bottleneck',_DenseBlock(bottleneck_layers, cur_channels_count, 
                                     bn_size, growth_rate, drop_rate))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels 

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)):
            self.transUpBlocks.append(_TransitionUp(cur_channels_count, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(_DenseBlock( up_blocks[i],
                cur_channels_count, bn_size, growth_rate, drop_rate,))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels


        #####################
        #      Softmax      #
        #####################

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count, 
               out_channels=n_classes, kernel_size=1, stride=1, 
                   padding=0, bias=True)

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

        if is_test:
            print("DBD size", out.size())
            
        out = self.bottleneck(out)

        if is_test:
            print ("bnecksize",out.size())
        
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            if is_test:
                print("DOWN_SKIP_PRE_UPSAMPLE",out.size(),skip.size())
            out = self.transUpBlocks[i](out, skip)
            if is_test:
                print("DOWN_SKIP_AFT_UPSAMPLE",out.size(),skip.size())
            out = self.denseBlocksUp[i](out)
        if is_test:
            print("after up", out.size(), skip.size())  

        out = self.finalConv(out)
        out = torch.squeeze(out, dim=1)
        
        return out

# Begin gross code :/
# Here's where we define the internals of the efficient bottleneck layer


class _EfficientDensenetBottleneckFn(Function):
    """
    The autograd function which performs the efficient bottlenck operations.
    Each of the sub-operations -- concatenation, batch normalization, ReLU,
    and convolution -- are abstracted into their own classes
    """
    def __init__(self, shared_allocation_1, shared_allocation_2,
            running_mean, running_var,
            stride=1, padding=0, dilation=1, groups=1,
            training=False, momentum=0.1, eps=1e-5):

        self.efficient_cat = _EfficientCat(shared_allocation_1.storage)
        self.efficient_batch_norm = _EfficientBatchNorm(shared_allocation_2.storage, running_mean, running_var,
                training, momentum, eps)
        self.efficient_relu = _EfficientReLU()
        self.efficient_conv = _EfficientConv2d(stride, padding, dilation, groups)

        # Buffers to store old versions of bn statistics
        self.prev_running_mean = self.efficient_batch_norm.running_mean.new()
        self.prev_running_mean.resize_as_(self.efficient_batch_norm.running_mean)
        self.prev_running_var = self.efficient_batch_norm.running_var.new()
        self.prev_running_var.resize_as_(self.efficient_batch_norm.running_var)
        self.curr_running_mean = self.efficient_batch_norm.running_mean.new()
        self.curr_running_mean.resize_as_(self.efficient_batch_norm.running_mean)
        self.curr_running_var = self.efficient_batch_norm.running_var.new()
        self.curr_running_var.resize_as_(self.efficient_batch_norm.running_var)


    def forward(self, bn_weight, bn_bias, conv_weight, *inputs):
        self.prev_running_mean.copy_(self.efficient_batch_norm.running_mean)
        self.prev_running_var.copy_(self.efficient_batch_norm.running_var)

        bn_input = self.efficient_cat.forward(*inputs)
        bn_output = self.efficient_batch_norm.forward(bn_weight, bn_bias, bn_input)
        relu_output = self.efficient_relu.forward(bn_output)
        conv_output = self.efficient_conv.forward(conv_weight, None, relu_output)

        self.bn_weight = bn_weight
        self.bn_bias = bn_bias
        self.conv_weight = conv_weight
        self.inputs = inputs
        return conv_output


    def backward(self, grad_output):
        # Turn off bn training status, and temporarily reset statistics
        training = self.efficient_batch_norm.training
        self.curr_running_mean.copy_(self.efficient_batch_norm.running_mean)
        self.curr_running_var.copy_(self.efficient_batch_norm.running_var)
        # self.efficient_batch_norm.training = False
        self.efficient_batch_norm.running_mean.copy_(self.prev_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.prev_running_var)

        # Recompute concat and BN
        cat_output = self.efficient_cat.forward(*self.inputs)
        bn_output = self.efficient_batch_norm.forward(self.bn_weight, self.bn_bias, cat_output)
        relu_output = self.efficient_relu.forward(bn_output)

        # Conv backward
        conv_weight_grad, _, conv_grad_output = self.efficient_conv.backward(
                self.conv_weight, None, relu_output, grad_output)

        # ReLU backward
        relu_grad_output = self.efficient_relu.backward(bn_output, conv_grad_output)

        # BN backward
        self.efficient_batch_norm.running_mean.copy_(self.curr_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.curr_running_var)
        bn_weight_grad, bn_bias_grad, bn_grad_output = self.efficient_batch_norm.backward(
                self.bn_weight, self.bn_bias, cat_output, relu_grad_output)

        # Input backward
        grad_inputs = self.efficient_cat.backward(bn_grad_output)

        # Reset bn training status and statistics
        self.efficient_batch_norm.training = training
        self.efficient_batch_norm.running_mean.copy_(self.curr_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.curr_running_var)

        return tuple([bn_weight_grad, bn_bias_grad, conv_weight_grad] + list(grad_inputs))


# The following helper classes are written similarly to pytorch autogrd functions.
# However, they are designed to work on tensors, not variables, and therefore
# are not functions.


class _EfficientBatchNorm(object):
    def __init__(self, storage, running_mean, running_var,
            training=False, momentum=0.1, eps=1e-5):
        self.storage = storage
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps

    def forward(self, weight, bias, input):
        # Assert we're using cudnn
        for i in ([weight, bias, input]):
            if i is not None and not(cudnn.is_acceptable(i)):
                raise Exception('You must be using CUDNN to use _EfficientBatchNorm')

        # Create save variables
        self.save_mean = self.running_mean.new()
        self.save_mean.resize_as_(self.running_mean)
        self.save_var = self.running_var.new()
        self.save_var.resize_as_(self.running_var)

        # Do forward pass - store in input variable
        res = type(input)(self.storage)
        res.resize_as_(input)
        torch._C._cudnn_batch_norm_forward(
            input, res, weight, bias, self.running_mean, self.running_var,
            self.save_mean, self.save_var, self.training, self.momentum, self.eps
        )

        return res

    def backward(self, weight, bias, input, grad_output):
        # Create grad variables
        grad_weight = weight.new()
        grad_weight.resize_as_(weight)
        grad_bias = bias.new()
        grad_bias.resize_as_(bias)

        # Run backwards pass - result stored in grad_output
        grad_input = grad_output
        torch._C._cudnn_batch_norm_backward(
            input, grad_output, grad_input, grad_weight, grad_bias,
            weight, self.running_mean, self.running_var, self.save_mean,
            self.save_var, self.training, self.eps
        )

        # Unpack grad_output
        res = tuple([grad_weight, grad_bias, grad_input])
        return res


class _EfficientCat(object):
    def __init__(self, storage):
        self.storage = storage

    def forward(self, *inputs):
        # Get size of new varible
        self.all_num_channels = [input.size(1) for input in inputs]
        size = list(inputs[0].size())
        for num_channels in self.all_num_channels[1:]:
            size[1] += num_channels

        # Create variable, using existing storage
        res = type(inputs[0])(self.storage).resize_(size)
        torch.cat(inputs, dim=1, out=res)
        return res

    def backward(self, grad_output):
        # Return a table of tensors pointing to same storage
        res = []
        index = 0
        for num_channels in self.all_num_channels:
            new_index = num_channels + index
            res.append(grad_output[:, index:new_index])
            index = new_index

        return tuple(res)


class _EfficientReLU(object):
    def __init__(self):
        pass

    def forward(self, input):
        backend = type2backend[type(input)]
        output = input
        backend.Threshold_updateOutput(backend.library_state, input, output, 0, 0, True)
        return output

    def backward(self, input, grad_output):
        grad_input = grad_output
        grad_input.masked_fill_(input <= 0, 0)
        return grad_input


class _EfficientConv2d(object):
    def __init__(self, stride=1, padding=0, dilation=1, groups=1):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def _output_size(self, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding
            kernel = self.dilation * (weight.size(d + 2) - 1) + 1
            stride = self.stride
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError("convolution input is too small (output would be {})".format(
                             'x'.join(map(str, output_size))))
        return output_size

    def forward(self, weight, bias, input):
        # Assert we're using cudnn
        for i in ([weight, bias, input]):
            if i is not None and not(cudnn.is_acceptable(i)):
                raise Exception('You must be using CUDNN to use _EfficientBatchNorm')

        res = input.new(*self._output_size(input, weight))
        self._cudnn_info = torch._C._cudnn_convolution_full_forward(
            input, weight, bias, res,
            (self.padding, self.padding),
            (self.stride, self.stride),
            (self.dilation, self.dilation),
            self.groups, cudnn.benchmark
        )

        return res

    def backward(self, weight, bias, input, grad_output):
        grad_input = input.new()
        grad_input.resize_as_(input)
        torch._C._cudnn_convolution_backward_data(
            grad_output, grad_input, weight, self._cudnn_info,
            cudnn.benchmark)

        grad_weight = weight.new().resize_as_(weight)
        torch._C._cudnn_convolution_backward_filter(grad_output, input, grad_weight, self._cudnn_info,
                                                    cudnn.benchmark)

        if bias is not None:
            grad_bias = bias.new().resize_as_(bias)
            torch._C._cudnn_convolution_backward_bias(grad_output, grad_bias, self._cudnn_info)
        else:
            grad_bias = None

        return grad_weight, grad_bias, grad_input

def FCDenseNet103(in_shape, n_classes=1):
    return FCDenseNet(in_channels=in_shape[0], down_blocks=(4,5,7,10,12),
                 up_blocks=(12,10,7,5,4), bottleneck_layers=15, bn_size=2,
                 growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)

def my_FCDenseNet(in_shape, n_classes=1):
    return FCDenseNet(in_channels=in_shape[0], down_blocks=(4, 4, 4, 4),
                 up_blocks=(4, 4, 4, 4), bottleneck_layers=5, 
                 growth_rate=12, out_chans_first_conv=48, n_classes=n_classes)

def my_FCDenseNet02(in_shape, n_classes=1):
    return FCDenseNet(in_channels=in_shape[0], down_blocks=(3,4,5,7,10),
                 up_blocks=(10,7,5,4,3), bottleneck_layers=15, bn_size=1,
                 growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)

def my_FCDenseNet03(in_shape, n_classes=1):
    return FCDenseNet(in_channels=in_shape[0], down_blocks=(2,3,5,7,10),
                 up_blocks=(10,7,5,3,2), bottleneck_layers=10, bn_size=1,
                 growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)
# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    CARVANA_HEIGHT = 1280
    CARVANA_WIDTH  = 1918
    batch_size  = 1
    #C,H,W = 3,1024,1024    #3,CARVANA_HEIGHT,CARVANA_WIDTH
    #C,H,W = 3,512,512
    #C,H,W = 3,640,960
    C,H,W = 3,704,1056
    #C,H,W = 3,768,1152

    if 1: # BCELoss2d()
        num_classes = 1

        inputs = torch.randn(batch_size,C,H,W)
        labels = torch.LongTensor(batch_size,H,W).random_(1).type(torch.FloatTensor)

        #net = FCDenseNet103(in_shape=(C,H,W)).cuda().train()
        net = my_FCDenseNet02(in_shape=(C,H,W)).cuda().train()
        print(type(net))
        print(net)

        x = Variable(inputs.cuda())
        y = Variable(labels.cuda())
        logits = net.forward(x,is_test=True)
        print(logits.size())

        loss = BCELoss2d()(logits, y)
        loss.backward()

        print('logits')
        print(logits)
    #input('Press ENTER to continue.')