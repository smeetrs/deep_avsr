import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .visual_frontend import VisualFrontend


class PositionalEncoding(nn.Module):

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).view(maxLen, 1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
        pe[:, 0::2] = torch.sin(position/denominator)
        pe[:, 1::2] = torch.cos(position/denominator)
        pe = pe.view(1, maxLen, dModel).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.size(0),:,:]
        return outputBatch



class AVNet(nn.Module):

    def __init__(self, dModel, nHeads, numLayers, peMaxLen, inSize, fcHiddenSize, dropout, numClasses):
        super(AVNet, self).__init__()
        self.audioConv = nn.Conv1d(inSize, dModel, kernel_size=4, stride=4, padding=0)
        self.frontend = VisualFrontend()
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.jointConv = nn.Conv1d(2*dModel, dModel, kernel_size=1, stride=1, padding=0)
        self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.outputConv = nn.Conv1d(dModel, numClasses, kernel_size=1, stride=1, padding=0)
        return

    def forward(self, inputBatch):
        audioInputBatch, videoInputBatch = inputBatch
        
        audioInputBatch = audioInputBatch.transpose(0, 1).transpose(1, 2)
        audioBatch = self.audioConv(audioInputBatch)
        audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
        audioBatch = self.positionalEncoding(audioBatch)
        audioBatch = self.audioEncoder(audioBatch)
        
        videoInputBatch = videoInputBatch.transpose(0, 1).transpose(1, 2)
        videoBatch = self.frontend(videoInputBatch)
        videoBatch = videoBatch.transpose(1, 2).transpose(0, 1)
        videoBatch = self.positionalEncoding(videoBatch)
        videoBatch = self.videoEncoder(videoBatch)

        jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
        jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
        jointBatch = self.jointConv(jointBatch)
        jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        jointBatch = self.jointDecoder(jointBatch)
        jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
        jointBatch = self.outputConv(jointBatch)
        outputBatch = F.log_softmax(jointBatch, dim=1)
        outputBatch = outputBatch.transpose(1, 2).transpose(0, 1)
        return outputBatch