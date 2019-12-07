import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class LRS2CharLM(nn.Module):

    def __init__(self):
        super(LRS2CharLM, self).__init__()
        self.embedding = nn.Embedding(38, 1024, padding_idx=None)
        self.lstm = nn.LSTM(1024, 1024, num_layers=4)
        self.fc = nn.Linear(1024, 38)
        return


    def forward(self, inputBatch, initStateBatch):
        batch = self.embedding(inputBatch)
        if initStateBatch != None:
            batch, finalStateBatch = self.lstm(batch, initStateBatch)
        else:
            batch, finalStateBatch = self.lstm(batch)
        batch = batch.transpose(0, 1)
        batch = F.log_softmax(self.fc(batch), dim=2)
        outputBatch = batch.transpose(0, 1)
        return outputBatch, finalStateBatch


