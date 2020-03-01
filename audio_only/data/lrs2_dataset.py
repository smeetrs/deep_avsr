import torch
from torch.utils.data import Dataset
import numpy as np

from .utils import prepare_pretrain_input
from .utils import prepare_main_input


class LRS2Pretrain(Dataset):
    
    def __init__(self, datadir, numWords, charToIx, stepSize, audioParams):
        super(LRS2Pretrain, self).__init__()
        with open(datadir + "/pretrain.txt", "r") as f:
            lines = f.readlines()
        self.datalist = [datadir + "/pretrain/" + line.strip() for line in lines]
        self.numWords = numWords
        self.charToIx = charToIx
        self.stepSize = stepSize
        self.audioParams = audioParams
        return
        

    def __getitem__(self, index):
        base = self.stepSize * np.arange(int(len(self.datalist)/self.stepSize)+1)
        ixs = base + index
        ixs = ixs[ixs < len(self.datalist)]
        index = np.random.choice(ixs)
        
        audioFile = self.datalist[index] + ".wav"
        targetFile = self.datalist[index] + ".txt"
        inp, trgt, inpLen, trgtLen = prepare_pretrain_input(audioFile, targetFile, self.numWords, self.charToIx, self.audioParams)
        return inp, trgt, inpLen, trgtLen


    def __len__(self):
        return self.stepSize




class LRS2Main(Dataset):
    
    def __init__(self, dataset, datadir, charToIx, stepSize, audioParams):
        super(LRS2Main, self).__init__()
        with open(datadir + "/" + dataset + ".txt", "r") as f:
            lines = f.readlines()
        self.datalist = [datadir + "/main/" + line.strip().split(" ")[0] for line in lines]
        self.charToIx = charToIx
        self.dataset = dataset
        self.stepSize = stepSize
        self.audioParams = audioParams
        return
        

    def __getitem__(self, index):
        if self.dataset == "train":
            base = self.stepSize * np.arange(int(len(self.datalist)/self.stepSize)+1)
            ixs = base + index
            ixs = ixs[ixs < len(self.datalist)]
            index = np.random.choice(ixs)

        audioFile = self.datalist[index] + ".wav"
        targetFile = self.datalist[index] + ".txt"
        inp, trgt, inpLen, trgtLen = prepare_main_input(audioFile, targetFile, self.charToIx, self.audioParams)
        return inp, trgt, inpLen, trgtLen


    def __len__(self):
        if self.dataset == "train":
            return self.stepSize
        else:
            return len(self.datalist)

