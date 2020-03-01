import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import cv2 as cv
from scipy.special import softmax



def prepare_main_input(visualFeaturesFile, targetFile, charToIx, videoParams):
    
    videoFPS = videoParams["videoFPS"]
    

    with open(targetFile, "r") as f:
        trgt = f.readline().strip()[7:]

    trgt = [charToIx[char] for char in trgt]
    trgt.append(charToIx["<EOS>"])
    trgt = np.array(trgt)
    trgtLen = len(trgt)

    if trgtLen > 256:
        print("Max target length reached. Exiting")
        exit()


    inp = np.load(visualFeaturesFile)

    reqInpLen = req_input_length(trgt)
    if len(inp) < reqInpLen:
        indices = np.arange(len(inp))
        np.random.shuffle(indices)
        repetitions = int((reqInpLen - len(inp))/len(inp)) + 1
        extras = (reqInpLen - len(inp)) % len(inp)
        newIndices = np.sort(np.concatenate([np.repeat(indices, repetitions), indices[:extras]]))
        inp = inp[newIndices]

    inpLen = len(inp)


    inp = torch.from_numpy(inp)
    trgt = torch.from_numpy(trgt)
    inpLen = torch.tensor(inpLen)
    trgtLen = torch.tensor(trgtLen)

    return inp, trgt, inpLen, trgtLen



def prepare_pretrain_input(visualFeaturesFile, targetFile, numWords, charToIx, videoParams):
    
    videoFPS = videoParams["videoFPS"]
    

    with open(targetFile, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    
    trgt = lines[0][7:]
    words = trgt.split(" ")


    if len(words) <= numWords:
        trgtNWord = trgt
        if len(trgtNWord)+1 > 256:
            print("Max target length reached. Exiting")
            exit()
        inp = np.load(visualFeaturesFile)

    else:
        nWords = [" ".join(words[i:i+numWords]) for i in range(len(words)-numWords+1)]
        nWordLens = np.array([len(nWord)+1 for nWord in nWords]).astype(np.float)
        nWordLens[nWordLens > 256] = -np.inf
        if np.all(nWordLens == -np.inf):
            print("Max target length reached. Exiting")
            exit()
        
        ix = np.random.choice(np.arange(len(nWordLens)), p=softmax(nWordLens))
        trgtNWord = nWords[ix]

        videoStartTime = float(lines[4+ix].split(" ")[1])
        videoEndTime = float(lines[4+ix+numWords-1].split(" ")[2])
        inp = np.load(visualFeaturesFile)
        inp = inp[int(np.floor(videoFPS*videoStartTime)):int(np.ceil(videoFPS*videoEndTime))]

    
    trgt = [charToIx[char] for char in trgtNWord]
    trgt.append(charToIx["<EOS>"])
    trgt = np.array(trgt)
    trgtLen = len(trgt)
    

    reqInpLen = req_input_length(trgt)
    if len(inp) < reqInpLen:
        indices = np.arange(len(inp))
        np.random.shuffle(indices)
        repetitions = int((reqInpLen - len(inp))/len(inp)) + 1
        extras = (reqInpLen - len(inp)) % len(inp)
        newIndices = np.sort(np.concatenate([np.repeat(indices, repetitions), indices[:extras]]))
        inp = inp[newIndices]

    inpLen = len(inp)
    

    inp = torch.from_numpy(inp)
    trgt = torch.from_numpy(trgt)
    inpLen = torch.tensor(inpLen)
    trgtLen = torch.tensor(trgtLen)
    
    return inp, trgt, inpLen, trgtLen



def collate_fn(dataBatch):
    inputBatch = pad_sequence([data[0] for data in dataBatch])
    targetBatch = torch.cat([data[1] for data in dataBatch])
    inputLenBatch = torch.stack([data[2] for data in dataBatch])
    targetLenBatch = torch.stack([data[3] for data in dataBatch])
    return inputBatch, targetBatch, inputLenBatch, targetLenBatch



def req_input_length(trgt):
    reqLen = len(trgt)
    lastChar = trgt[0]
    for i in range(1, len(trgt)):
        if trgt[i] != lastChar:
            lastChar = trgt[i]
        else:
            reqLen = reqLen + 1
    return reqLen

