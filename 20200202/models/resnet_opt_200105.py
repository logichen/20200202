'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
import pandas as pd

#
# class sine(nn.Module):
#     def __init__(self):
#         super(sine, self).__init__()
#
#     def forward(self, x):
#         x = torch.sin(x)
#         return x


# class sine(nn.Module):
#     def __init__(self):
#         super(sine, self).__init__()
#
#     def forward(self, x):
#         x = x * F.sigmoid(x)
#         return x

# def sine(x):
#     x = torch.sin(x)
#     return x

def sine(x):
    x = torch.exp(- torch.pow(x, 2))
    return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) #in_planes: num of input channels,plames:num of output channels
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.elu(self.bn1(self.conv1(x)))
        # out =sine(self.bn1(self.conv1(x)))
        # out = F.leaky_sin(self.bn1(self.conv1(x)), negative_slope=0.1) # layer 1: x.shape is 256,64,32.32, out2.shape is 256,64,32.32  #there is a second step,x is reshaped. layer2:out2.shape is 256,128,16.16, x.shape is 256,128,16.16,  #layer3:out2.shape is 256,256,8.8, x.shape is 256,256,8,8， #layer4:out2.shape is 256,512,4.4, x.shape is 256,256,8,8
        # print('x', x.shape, '\n')
        # print('weight', net.module.conv1.weight.shape,'\n')
        # print('out', out.shape,'\n')
        # print(np.dot(np.linalg.pinv(x.data.cpu().numpy()), x.data.cpu().numpy() ))
        out = self.bn2(self.conv2(out)) # layer1:  out2.shape is 256,64,32.32,# layer2:  out2.shape is 256,128,16,16,#layer 3: out2.shape=256.256,8,8 ##layer4:out2.shape is 256,512,4.4
        out += self.shortcut(x) #layer1: out2.shape is 256,64,32.32 #layer2: out2.shape is 256,128,16,16 #layer3: out2.shape is 256,256,8,8， #layer4:out2.shape is 256,512,4.4
        out =F.relu(out)
        # out = F.leaky_sin(out, negative_slope=0.1)# layer 1: out2.shape is 256,64,32.32, #layer 2: out2.shape is 256,128.16,16, #layer3: out2.shape is 256,256,8,8， #layer4:out2.shape is 256,512,4.4
        return out
'''
    # backward begin
    def backward(self, x, y):

    # backward end
'''

class Bottleneck(nn.Module):
    expansion = 4 #expansion is used to give some copies with different Brightness
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out =F.relu(self.bn1(self.conv1(x)))
        # out =F.gelu(self.bn1(self.conv1(x)))

        # out =sine(self.bn1(self.conv1(x)))
        # out = F.leaky_sin(self.bn1(self.conv1(x)), negative_slope=0.1)
        out =F.relu(self.bn2(self.conv2(out)))
        # out =F.tanh(self.bn2(self.conv2(out)))

        # out =sine(self.bn2(self.conv2(out)))
        # out = F.leaky_sin(self.bn2(self.conv2(out)), negative_slope=0.1)

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        # out =sine(out)
        # out = F.leaky_sin(out, negative_slope=0.1)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) # what is initial value of num_blocks？
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)



    def _make_layer(self, block, planes, num_blocks, stride): #planes = num of ouput channels
        strides = [stride] + [1]*(num_blocks-1) #??
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def _make_layer(self, block, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion: #if dims are not match
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             nn.BatchNorm2d(planes * block.expansion),
    #         )




    def forward(self, x):
        out =F.relu(self.bn1(self.conv1(x)))

        # out =F.gelu(self.bn1(self.conv1(x)))

        # out =sine(self.bn1(self.conv1(x)))
        # out = F.leaky_sin(self.bn1(self.conv1(x)), negative_slope=0.1) # x.shape is 256,3,32.32, # out2.shape is 256,64,32.32

        out = self.layer1(out) # out2.shape is 256,64,32.32,
        out = self.layer2(out) # out2.shape is 256,128,16.16,
        out = self.layer3(out) # out2.shape is 256,256,8.8,
        out = self.layer4(out) # out2.shape is 256,512,4.4
        out = F.avg_pool2d(out, 4) # out2.shape is 256,512,1,1
        out = out.view(out.size(0), -1) #trans to vector. # out2.shape is 256,512
        # out_mocpnn = out.clone()
        # print('out:\n', out)

        # out_save = out.data.cpu().numpy()
        # out_save = out_save.tolist()
        # out_save = pd.DataFrame(data= out_save)
        # # if not os.path.isdir('out'):
        # #     os.mkdir('out')
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir("/home/ps/media/ps/lab415/clm/Code_415_ubuntu/Pytorch/pytorch-cifar-master")
        # out_save.to_csv("/home/ps/media/ps/lab415/clm/Code_415_ubuntu/Pytorch/pytorch-cifar-master/out_save.csv", mode='a+', index=None, header=None)
        '''could be modified by SOCPNN, maybe insert a layer?'''
        out = self.linear(out) # out2.shape is 256,10

        # out1 = self.linear1(out) # out2.shape is 256,10
        # out2 = self.linear2(out) # out2.shape is 256,10
        # out3 = self.linear3(out) # out2.shape is 256,10
        # out4 = self.linear4(out) # out2.shape is 256,10
        # out5 = self.linear5(out) # out2.shape is 256,10
        #
        # out = self.poly1(out1) + self.poly2(out2) + self.poly3(out3) + self.poly4(out4) + self.poly5(out5)
        # return out, out_mocpnn
        return out

'''
#backward begin
    def backward(self, y, x):

        return parameters
#backward end
'''
# def ResNet18():
def ResNet18_opt_200105():
    return ResNet(BasicBlock, [2,2,2,2])
    # print(Resnet.layer1.shape)


def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3]) # layer1 consist 3 blocks, layer2 consist 4 blocks, layer3 consist 6 blocks...

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18_leaky_1225()
    y = net(torch.randn(2, 3,32,32))
    print(y.size())

# test()



#
# dummy_input = torch.rand(13, 3, 64, 64)
# model = ResNet18_leaky_0924()
# with SummaryWriter(comment='LeNet') as w:
#     w.add_graph(model, (dummy_input,))



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = ResNet18_leaky_0924()
#
# net = net.to(device)
#
# summary(net, (3, 32, 32))