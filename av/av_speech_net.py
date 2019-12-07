import math
import torch
import torch.nn
import torch.nn.functional as F
from visual_frontend import VisualFrontend



class PositionalEncoding(nn.Module):

    def __init__(self, dModel=512, maxLen=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).view(maxLen, 1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
        pe[:, 0::2] = torch.sin(position/denominator)
        pe[:, 1::2] = torch.cos(position/denominator)
        pe = pe.view(1, maxLen, dModel).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0),:,:]
        return x





class AVSpeechNet(nn.Module):

    def __init__(self, dModel=512, nHeads=8, numLayers=6):
        super(AVSpeechNet, self).__init__()
        self.visualFrontend = VisualFrontend()
        self.positionalEncoding = PositionalEncoding(maxLen=5000)
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=2048, dropout=0.1)
        self.visualEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.audioConv = nn.Conv1d(321, dModel, kernel_size=4, stride=1, padding=0)
        self.jointConv = nn.Conv1d(2*dModel, dModel, kernel_size=1, stride=1, padding=0)
        return

    def forward(self, x):
        return x



if __name__ == '__main__':
    exit()