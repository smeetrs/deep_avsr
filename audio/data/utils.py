import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.special import softmax
import sounddevice as sd
import time



def prepare_main_input(audioFile, targetFile, charToIx, stftParams):
    
    with open(targetFile, "r") as f:
        trgt = f.readline().strip()[7:]
    
    sampFreq, inputAudio = wavfile.read(audioFile)
    inputAudio = inputAudio/np.max(inputAudio)
    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))


    trgt = np.array([charToIx[char] for char in trgt])
    trgtLen = len(trgt)

    stftWindow = stftParams["window"]
    stftWinLen = stftParams["winLen"]
    stftOverlap = stftParams["overlap"]    
    _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq*stftWinLen, 
                                 noverlap=sampFreq*stftOverlap, boundary=None, padded=False)
    inp = np.abs(stftVals)
    inp = inp.T


    reqInpLen = req_input_length(trgt)
    if int(len(inp)/4) < reqInpLen:
        indices = np.arange(len(inp))
        np.random.shuffle(indices)
        repetitions = int(((reqInpLen - int(len(inp)/4))*4)/len(inp)) + 1
        extras = ((reqInpLen - int(len(inp)/4))*4) % len(inp)
        newIndices = np.sort(np.concatenate((np.repeat(indices, repetitions), indices[:extras])))
        inp = inp[newIndices]

    inpLen = int(len(inp)/4)

    inp = torch.from_numpy(inp)
    trgt = torch.from_numpy(trgt)
    inpLen = torch.tensor(inpLen)
    trgtLen = torch.tensor(trgtLen)

    return inp, trgt, inpLen, trgtLen



def prepare_pretrain_input(audioFile, targetFile, numWords, charToIx, stftParams):
    
    with open(targetFile, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    
    trgt = lines[0][7:]
    words = trgt.split(" ")


    if len(words) <= numWords:
        trgtNWord = trgt
        if len(trgtNWord) > 256:
            print("Max target length reached. Exiting")
            exit()
        sampFreq, inputAudio = wavfile.read(audioFile)

    else:
        nWords = [" ".join(words[i:i+numWords]) for i in range(len(words)-numWords+1)]
        nWordLens = np.array([len(nWord) for nWord in nWords]).astype(np.float)
        nWordLens[nWordLens > 256] = -np.inf
        if np.all(nWordLens == -np.inf):
            print("Max target length reached. Exiting")
            exit()
        
        ix = np.random.choice(np.arange(len(nWordLens)), p=softmax(nWordLens))
        trgtNWord = nWords[ix]
        
        audioStartTime = float(lines[4+ix].split(" ")[1])
        audioEndTime = float(lines[4+ix+numWords-1].split(" ")[2])
        sampFreq, audio = wavfile.read(audioFile)
        inputAudio = audio[int(sampFreq*audioStartTime):int(sampFreq*audioEndTime)]

    inputAudio = inputAudio/np.max(inputAudio)

    stftWindow = stftParams["window"]
    stftWinLen = stftParams["winLen"]
    stftOverlap = stftParams["overlap"]
    if len(inputAudio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
        padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(inputAudio))/2))
        inputAudio = np.pad(inputAudio, padding, "constant")
    
    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))


    trgt = np.array([charToIx[char] for char in trgtNWord])
    trgtLen = len(trgt)


    _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq*stftWinLen, 
                                 noverlap=sampFreq*stftOverlap, boundary=None, padded=False)
    inp = np.abs(stftVals)
    inp = inp.T

    reqInpLen = req_input_length(trgt)
    if int(len(inp)/4) < reqInpLen:
        indices = np.arange(len(inp))
        np.random.shuffle(indices)
        repetitions = int(((reqInpLen - int(len(inp)/4))*4)/len(inp)) + 1
        extras = ((reqInpLen - int(len(inp)/4))*4) % len(inp)
        newIndices = np.sort(np.concatenate((np.repeat(indices, repetitions), indices[:extras])))
        inp = inp[newIndices]

    inpLen = int(len(inp)/4)

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

