#importing the required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#the basic block used in the resnet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = out + residual
        out = F.relu(out)
        return out



#constructs the resnet
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
            
        
    #constructs a layer given the block to use, output channels, block repetitions and the net stride
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


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x






#visual frontend for extracting visual features
class VisualFrontend(nn.Module):

    def __init__(self):
        super(VisualFrontend, self).__init__()
        
        #the 3D frontend
        self.frontend3D = nn.Sequential(
                            nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                            nn.BatchNorm3d(64),
                            nn.ReLU(),
                            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
                        )
        #resnet
        self.resnet = ResNet()
        return


    def forward(self, x):
        batchSize = x.size(0)
        x = self.frontend3D(x)

        #reshape (N,C,T,H,W) to (N*T,C,H,W)
        x = x.transpose(1, 2)
        x = x.view(-1, 64, x.size(3), x.size(4))

        x = self.resnet(x)
        
        #revert the reshape to get a 512-dimensional feature vector for each frame
        x = x.view(batchSize, -1, 512)
        x = x.transpose(0, 1)
        return x
        


        
if __name__=="__main__":
    
    #testing the VisualFrontend module
    import cv2 as cv

    #read the mouth ROI sequence image and reshape it to (N,C,T,H,W)
    #N -> Batch size, C -> No of Channels, T -> Depth, H -> Height, W-> Width
    #convert the pixel values to lie between 0 and 1
    img = cv.imread("./demo/00004_roi.png", 0)
    ch = np.split(img, img.shape[1]/112, axis=1)
    ch = [elm.reshape((112,112,1)) for elm in ch]
    inp = np.dstack(ch)
    inp = np.transpose(inp, (2,0,1))
    inp = inp.reshape((1,1,inp.shape[0],112,112))
    inp = torch.from_numpy(inp)
    inp = inp.float()
    inp = inp/255
    
    #pass the input to the VisualFrontend and obtain the output
    vf = VisualFrontend().to("cpu")
    vf.eval()
    out = vf.forward(inp)
    print(out.size())