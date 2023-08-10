import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from kymatio.torch import Scattering2D  # ellen for scattering
from torch.nn.functional import interpolate
import torch.nn.functional as F # ellen - carafe
import torch
import pdb  #ellen

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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

class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class scatter_transform(nn.Module):
    def __init__(self, in_channels, out_channels, size, level):
        super(scatter_transform, self).__init__()
        self.output_size = int(size/(2**level))
        input_size = int(size/(2**(level-1)))
        self.input_size = input_size
        J = 1
        self.Scattering = Scattering2D(
            J, (input_size, input_size)).cuda()  # cuda 추가!
        self.conv1x1 = nn.Conv2d(
            in_channels*9, out_channels, kernel_size=1, padding=0)


    def forward(self, x):
        scatter_ouput = self.Scattering.scattering(x)
        scatter_ouput = scatter_ouput.view(scatter_ouput.size(
            0), -1, self.output_size, self.output_size)
        scatter_ouput = interpolate(scatter_ouput, size=(self.input_size,self.input_size), mode = "bilinear")
        scatter_ouput = self.conv1x1(scatter_ouput)
        return scatter_ouput


class scatter_transform_max(nn.Module):
    def __init__(self, in_channels, out_channels, size, level):
        super(scatter_transform_max, self).__init__()
        self.output_size = int(size/(2**level))
        input_size = int(size/(2**(level-1)))
        self.input_size = input_size
        J = 1
        self.Scattering = Scattering2D(
            J, (input_size, input_size)).cuda()  # cuda 추가!
        self.conv1x1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0)


    def forward(self, x):
        scatter_ouput = self.Scattering.scattering(x)
        scatter_ouput=scatter_ouput[:,:,1:,:,:]  #channel who pass thr only low pass eliminated
        scatter_ouput = torch.max(scatter_ouput,dim=2).values # max

        scatter_ouput = interpolate(scatter_ouput, size=(self.input_size,self.input_size), mode = "bilinear")
        scatter_ouput = self.conv1x1(scatter_ouput)
        return scatter_ouput

class Bottleneck_scattering_ellen(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None,input_size=129):
        super(Bottleneck_scattering_ellen, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.scattering_down_1 = scatter_transform(inplanes,16, 129,1)


        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)

        self.conv_sc=nn.Conv2d(planes+16, planes, kernel_size=1)


        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)



        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        scattering1 = self.scattering_down_1(x)


        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu(out)

        out2 = torch.cat([out, scattering1], 1)
        out2= self.conv_sc(out2)
        out2 = self.relu(out2)

        out = self.conv3(out2)
        out = self.bn3(out)
        out3 = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out3 += residual
        out3 = self.relu(out3)

        return out3


class Bottleneck_scattering_ellen_max(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None,input_size=129):
        super(Bottleneck_scattering_ellen_max, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.scattering_down_1 = scatter_transform_max(inplanes,16, 129,1)


        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)

        self.conv_sc=nn.Conv2d(planes+16, planes, kernel_size=1)


        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)



        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        scattering1 = self.scattering_down_1(x)


        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu(out)

        out2 = torch.cat([out, scattering1], 1)
        out2= self.conv_sc(out2)
        out2 = self.relu(out2)

        out = self.conv3(out2)
        out = self.bn3(out)
        out3 = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out3 += residual
        out3 = self.relu(out3)

        return out3

class ResNet_scattering_ellen(nn.Module): 

    # Made by :  Ellen
    # Date: 2022.12.15
    # Base : Resnet_CARAFE

    # Brief summary: scattering transform while extracting skip connection features -> only one skip connection   

    def __init__(self, block, sf_block, layers, output_stride, BatchNorm, pretrained=True,input_size=129 ):
        # default  Bottelneck, [3, 4, 23, 3], output_stride:16, batchnorm: SynchronizedBatchNorm2d 
        self.inplanes = 64
        super(ResNet_scattering_ellen, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2] # dilation == 1 : no dilation
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError
    
        # ellen -dilation 
        dilations_ellen = [1, 6, 12, 18]
        # print(dilations_ellen)
        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, sf_block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm,input_size=129, use_scattering=True)
        self.layer2 = self._make_layer(block, sf_block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm,input_size=129, use_scattering=False)
        self.layer3 = self._make_layer(block, sf_block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm,input_size=129, use_scattering=False)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, sf_block, planes, blocks, stride=1, dilation=1, BatchNorm=None, dilation2= [1, 6, 12, 18],input_size=129, use_scattering = False):
        # block: Bottelneck, planes: 64/128/256/512, blocks: 3/4/23/3
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # block.expansion: 
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        if use_scattering: 
            for i in range(1, blocks-1): # ellen scattering-1,2 
                layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
            layers.append(sf_block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm, input_size=input_size))

            return nn.Sequential(*layers)
        else: 
            for i in range(1, blocks): #layers: 3,4,23,3
                layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
            return nn.Sequential(*layers)


    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        scattering_out=x

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)
        return x, scattering_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class ResNet_scattering_ellen3(nn.Module): 

    # Made by :  Ellen
    # Date: 2022.12.15
    # Base : Resnet_CARAFE

    # Brief summary: scattering transform while extracting skip connection features -> only one skip connection   

    def __init__(self, block, sf_block, layers, output_stride, BatchNorm, pretrained=True,input_size=129 ):
        # default  Bottelneck, [3, 4, 23, 3], output_stride:16, batchnorm: SynchronizedBatchNorm2d 
        self.inplanes = 64
        super(ResNet_scattering_ellen3, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2] # dilation == 1 : no dilation
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError
    
        # ellen -dilation 
        dilations_ellen = [1, 6, 12, 18]
        # print(dilations_ellen)
        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, sf_block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm,input_size=129, use_scattering=True)
        self.layer2 = self._make_layer(block, sf_block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm,input_size=129, use_scattering=False)
        self.layer3 = self._make_layer(block, sf_block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm,input_size=129, use_scattering=False)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, sf_block, planes, blocks, stride=1, dilation=1, BatchNorm=None, dilation2= [1, 6, 12, 18],input_size=129, use_scattering = False):
        # block: Bottelneck, planes: 64/128/256/512, blocks: 3/4/23/3
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # block.expansion: 
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        if use_scattering: 
            for i in range(1, blocks): # ellen scattering-3 
                layers.append(sf_block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm, input_size=input_size))

            return nn.Sequential(*layers)
        else: 
            for i in range(1, blocks): #layers: 3,4,23,3
                layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
            return nn.Sequential(*layers)


    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        scattering_out=x

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)
        return x, scattering_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet101(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
    # model = ResNet_scattering_ellen(Bottleneck, Bottleneck_scattering_ellen,[3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained, input_size = 129) # ellen 2022.12.14 - scattering add
    # model = ResNet_scattering_ellen3(Bottleneck, Bottleneck_scattering_ellen,[3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained, input_size = 129) # ellen 2022.12.14 - scattering add
    # model = ResNet_scattering_ellen(Bottleneck, Bottleneck_scattering_ellen_max,[3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained, input_size = 129) # ellen 2022.12.14 - scattering add
    # model = ResNet_scattering_ellen3(Bottleneck, Bottleneck_scattering_ellen_max,[3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained, input_size = 129) # ellen 2022.12.14 - scattering add    
    
    return model

if __name__ == "__main__":
    import torch
    model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)

    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())