import math
import torch
from torch.nn import ReplicationPad3d
import torch.nn as nn
from . import inflate


class InflatedResNet(torch.nn.Module):
    def __init__(self, 
                 resnet2d, 
                 frame_nb=4, 
                 class_nb=1000, 
                 conv_class=True):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(InflatedResNet, self).__init__()
        self.conv_class = conv_class

        self.conv1 = inflate.inflate_conv(
            resnet2d.conv1, time_dim=1, time_padding=0, center=True)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        
        self.layer3_b = inflate_reslayer(resnet2d.layer3_b)
        self.layer4 = inflate_reslayer(resnet2d.layer4)

        if conv_class:
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
            self.classifier = torch.nn.Conv3d(
                in_channels=2048,
                out_channels=51,
                kernel_size=(1, 1, 1),
                bias=True)

        else:
            final_time_dim = 1
            self.avgpool = inflate.inflate_pool(
                resnet2d.avgpool, time_dim=final_time_dim)
            self.fc = inflate.inflate_linear(resnet2d.fc, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
    
        x = self.layer3(x)

        if x.size()[2] == 4:
            x_1 = x

            x_1 = self.layer3_b(x_1)

            x_1 = self.layer4(x_1)

            if self.conv_class:
                x_1 = self.avgpool(x_1)
                x_1 = self.classifier(x_1)
                x_1 = x_1.squeeze(3)
                x_1 = x_1.squeeze(3)
                x_1 = x_1.mean(2)
            else:
                x_1 = self.avgpool(x_1)
                x_reshape = x_1.view(x_1.size(0), -1)
                x_1 = self.fc(x_reshape)

            return x, x_1
        
        else:

            return x


def inflate_reslayer(reslayer2d):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]
        #print("Spatial stride:", spatial_stride)

        self.conv1 = inflate.inflate_conv(
            bottleneck2d.conv1, time_dim=1, center=True)

        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate.inflate_conv(
            bottleneck2d.conv2,
            time_dim=1,
            time_padding=0,
            time_stride=1,
            center=True)

        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate.inflate_conv(
            bottleneck2d.conv3, time_dim=1, center=True)

        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=1)
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        inflate.inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=True),
        inflate.inflate_batch_norm(downsample2d[1]))
    return downsample3d
