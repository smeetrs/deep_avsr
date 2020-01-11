import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.downsample = downsample

    def forward(self, inputBatch):
        batch = F.relu(self.bn1(self.conv1(inputBatch)))
        batch = self.bn2(self.conv2(batch))
        if self.downsample is not None:
            residualBatch = self.downsample(inputBatch)
        else:
            residualBatch = inputBatch
        batch = batch + residualBatch
        outputBatch = F.relu(batch)
        return outputBatch




class ResNet(nn.Module):

    def __init__(self, repetitions=[2,2,2,2]):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(BasicBlock, 64, repetitions[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, repetitions[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, repetitions[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, repetitions[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))
        self.weights_init()
        return
            
        
    def _make_layer(self, block, outplanes, reps, stride):
        downsample = None
        if stride != 1 or self.inplanes != outplanes*block.expansion:
            downsample = nn.Sequential(
                            nn.Conv2d(self.inplanes, outplanes*block.expansion, kernel_size=(1,1), stride=stride, bias=False),
                            nn.BatchNorm2d(outplanes*block.expansion)
                        )

        layers = []
        layers.append(block(self.inplanes, outplanes, stride, downsample))
        self.inplanes = outplanes*block.expansion
        for i in range(reps-1):
            layers.append(block(self.inplanes, outplanes))

        return nn.Sequential(*layers)


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, np.sqrt(2/n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return


    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch




class VisualFrontend(nn.Module):

    def __init__(self):
        super(VisualFrontend, self).__init__()
        self.frontend3D = nn.Sequential(
                            nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                            nn.BatchNorm3d(64),
                            nn.ReLU(),
                            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
                        )
        self.resnet = ResNet()
        return


    def forward(self, inputBatch):
        batchsize = inputBatch.size(0)
        batch = self.frontend3D(inputBatch)

        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.size(0)*batch.size(1), batch.size(2), batch.size(3), batch.size(4))
        outputBatch = self.resnet(batch)
        outputBatch = outputBatch.view(batchsize, -1, 512)
        outputBatch = outputBatch.transpose(1 ,2)
        return outputBatch
        
