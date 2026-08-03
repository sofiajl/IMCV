## Network and its Parameters

import numpy as np
import torch
from torchvision import models
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

'''
model = DORN()
mapping = OrdinalMapping()
ordinal_optimizer = AdamW(DORN.parameters())
depth_optimizer = AdamW(mapping.parameters())

for epoch
  for imgs, depths
    preds_ordinal = model(imgs)
    probs = ordinal_method.to_probabilities(preds_ordinal)
    preds_depths = mapping(probs)

    loss_ordinal = ordinal_method.compute_loss(preds_ordinal) + spatial_regs(probs)
    loss_mapping = torch.mean((preds_depths - depths)**2)
    ordinal_optimizer.zero_grad()
    loss_ordinal.backward()
    ordinal_optimizer.step()
    mapping_optimizer.zero_grad()
    loss_mapping.backward()
    mapping_optimizer.step()

    # avaliar com mapping (preds_depths)
    # avaliar sem mapping (probs)
'''

# (1) Celso: nossas metricas
# (2) Jaime: ordinal mapping
# (3) Rafael: adicionar


# implemented using Torchvision
class DeepLab(torch.nn.Module):
    def __init__(self, K, backbone):
        super().__init__()
        assert backbone in ('MobileNetV3Large', 'ResNet50', 'ResNet101')

        if backbone == 'ResNet50':
            self.model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
            self.model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, K) # changes the last layer of the pre-trained model to output our n_outputs
        elif backbone == 'ResNet101':
            self.model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT')
            self.model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, K)
        else:
            self.model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
            self.model.classifier = models.segmentation.deeplabv3.DeepLabHead(960, K)

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)['out']

class FCN(torch.nn.Module):
    def __init__(self, K, backbone):
        super().__init__()
        assert backbone in ('ResNet50', 'ResNet101')

        if backbone == 'ResNet50':
            self.model = models.segmentation.fcn_resnet50(weights='DEFAULT')
            self.model.classifier = models.segmentation.fcn.FCNHead(2048, K)
        else:
            self.model = models.segmentation.fcn_resnet101(weights='DEFAULT')
            self.model.classifier = models.segmentation.fcn.FCNHead(2048, K)

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)['out']

class LRASPP(torch.nn.Module):
    def __init__(self, K):
        super().__init__()

        self.model = models.segmentation.lraspp_mobilenet_v3_large(weights='DEFAULT')
        self.model.classifier = models.segmentation.lraspp.LRASPPHead(low_channels=40, high_channels=960, num_classes=K, inter_channels=128)

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)['out']



# architecture presented in "Deep Ordinal Regression Network for Monocular Depth Estimation"
# credits to https://github.com/hufu6371/DORN

class FullImageEncoder(nn.Module):
    def __init__(self, h, w, kernel_size, stride, padding):
        super().__init__()

        self.input_height = int(np.floor((h - kernel_size + 2*padding) / stride) + 1)
        self.output_width = int(np.floor((w - kernel_size + 2*padding) / stride) + 1)

        self.global_pooling = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.dropout = nn.Dropout2d(p=0.5)
        self.global_fc = nn.Linear(2048*self.input_height*self.output_width, 512) # self.global_fc = nn.LazyLinear(512)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(512, 512, 1)

    def forward(self, x):
        x1 = self.global_pooling(x)
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048*self.input_height*self.output_width)
        x4 = self.relu(self.global_fc(x3))
        x4 = x4.view(-1, 512, 1, 1) # batch_size is inferred, 512 channels and spatial dimension 1x1
        x5 = self.relu(self.conv(x4))
        return x5

class SceneUnderstandingModule(nn.Module):
    def __init__(self, K, image_h, image_w, features_h, features_w, pyramid=[6, 12, 18]):
        super().__init__()

        assert len(pyramid) == 3
        self.image_h = image_h
        self.image_w = image_w
        self.features_h = features_h
        self.features_w = features_w

        # The full-image encoder captures global contextual information and can greatly clarify local confusions in depth estimation
        self.encoder = FullImageEncoder(h=features_h, w=features_w, kernel_size=16, stride=16, padding=16//2)

        # ASPP is employed to extract features from multiple large receptive fields via dilated convolutional operations
        self.aspp1 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                                nn.ReLU(inplace=True))

        self.aspp2 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=pyramid[0], dilation=pyramid[0], bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                                nn.ReLU(inplace=True))

        self.aspp3 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=pyramid[1], dilation=pyramid[1], bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                                nn.ReLU(inplace=True))

        self.aspp4 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=pyramid[2], dilation=pyramid[2], bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                                nn.ReLU(inplace=True))

        # Concatenate and apply conv layer
        self.concat_process = nn.Sequential(nn.Dropout2d(p=0.5),
                                        nn.Conv2d(512*5, 2048, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout2d(p=0.5),
                                        nn.Conv2d(2048, K, 1))

    def forward(self, x):
        x1 = self.encoder(x)
        x1 = F.interpolate(x1, size=(self.features_h, self.features_w), mode="bilinear", align_corners=True)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)
        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.concat_process(x6)
        out = F.interpolate(out, size=(self.image_h, self.image_w), mode="bilinear", align_corners=True)
        return out

class DORN_ResNet101(torch.nn.Module):
    def __init__(self, K, input_size=(385, 513), pyramid=[6, 12, 18]):
        super().__init__()

        self.image_h, self.image_w = input_size[0], input_size[1]

        resnet = models.resnet101(weights='DEFAULT') # pretrained on ImageNet

        resnet.conv1.requires_grad_(False)
        resnet.layer1.requires_grad_(False)

        for name, module in resnet.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-2]) # remove avgpool and fc resnet101 layers

        self.dummy_input = torch.randn(1, 3, self.image_h, self.image_w)
        with torch.no_grad():
            self.dummy_features = self.backbone(self.dummy_input)
        _, _, self.features_h, self.features_w = self.dummy_features.shape

        self.SceneUnderstandingModule = SceneUnderstandingModule(K, self.image_h, self.image_w, self.features_h, self.features_w, pyramid)

    def forward(self, image):
        feat = self.backbone(image)
        out = self.SceneUnderstandingModule(feat)
        return out


# architecture presented in "Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps with Accurate Object Boundaries"
# credits to https://github.com/JunjH

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                                    nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], 'pretrained_model/encoder'))
    return model

def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class UpProjection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)
        self.conv2 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))
        out = self.relu(bran1 + bran2)
        return out

class E_resnet(nn.Module): # encoder
    def __init__(self, original_model, num_features = 2048):
        super(E_resnet, self).__init__()        
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)
        return x_block1, x_block2, x_block3, x_block4

class D(nn.Module): # decoder
    def __init__(self, num_features = 2048):
        super(D, self).__init__()
        self.conv = nn.Conv2d(num_features, num_features // 2, kernel_size=1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = nn.BatchNorm2d(num_features)

        self.up1 = UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up2 = UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up3 = UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up4 = UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

    def forward(self, x_block1, x_block2, x_block3, x_block4, input_size):
        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d1 = self.up1(x_d0, [input_size[0] // 16, input_size[1] // 16])
        x_d2 = self.up2(x_d1, [input_size[0] // 8, input_size[1] // 8])
        x_d3 = self.up3(x_d2, [input_size[0] // 4, input_size[1] // 4])
        x_d4 = self.up4(x_d3, [input_size[0] // 2, input_size[1] // 2])
        return x_d4

class MFF(nn.Module): # multi-scale feature fusion
    def __init__(self, block_channel, num_features=64):
        super(MFF, self).__init__()

        self.up1 = UpProjection(num_input_features=block_channel[0], num_output_features=16)
        self.up2 = UpProjection(num_input_features=block_channel[1], num_output_features=16)
        self.up3 = UpProjection(num_input_features=block_channel[2], num_output_features=16)
        self.up4 = UpProjection(num_input_features=block_channel[3], num_output_features=16)
        self.conv = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, input_size):
        x_m1 = self.up1(x_block1, input_size)
        x_m2 = self.up2(x_block2, input_size)
        x_m3 = self.up3(x_block3, input_size)
        x_m4 = self.up4(x_block4, input_size)
        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)
        return x

class R(nn.Module):
    def __init__(self, block_channel, K):
        super(R, self).__init__()

        num_features = 64 + block_channel[3]//32
        print("num_features:", num_features)
        self.conv0 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features, K, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)
        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x2 = self.conv2(x1)
        return x2

class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel, K):
        super(model, self).__init__()
        self.E = Encoder
        self.D = D(num_features)
        self.MFF = MFF(block_channel)
        self.R = R(block_channel, K)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4, [x.size(2), x.size(3)])
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))
        out = self.final_upsample(out) # upsample to match the input size
        return out


class OrdinalMapping(torch.nn.Module):
    def __init__(self, K):
        super().__init__()
        # self.ordinal_loss = ordinal_loss
        self.fc = torch.nn.Linear(K, 1)

    def forward(self, x):
        # x = self.ordinal_loss.to_probabilities(x)
        x = self.fc(x)
        return x[:, 0]
