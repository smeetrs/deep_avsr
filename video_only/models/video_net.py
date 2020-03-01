import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PositionalEncoding(nn.Module):

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
        pe[:, 0::2] = torch.sin(position/denominator)
        pe[:, 1::2] = torch.cos(position/denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0],:,:]
        return outputBatch



class VideoNet(nn.Module):

    def __init__(self, dModel, nHeads, numLayers, peMaxLen, fcHiddenSize, dropout, numClasses):
        super(VideoNet, self).__init__()
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.videoDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.outputConv = nn.Conv1d(dModel, numClasses, kernel_size=1, stride=1, padding=0)
        return

    def forward(self, inputBatch):
        batch = self.positionalEncoding(inputBatch)
        batch = self.videoEncoder(batch)
        batch = self.videoDecoder(batch)
        batch = batch.transpose(0, 1).transpose(1, 2)
        batch = self.outputConv(batch)
        batch = batch.transpose(1, 2).transpose(0, 1)
        outputBatch = F.log_softmax(batch, dim=2)
        return outputBatch