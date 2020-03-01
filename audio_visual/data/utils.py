import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy import signal
from scipy.io import wavfile
import cv2 as cv
from scipy.special import softmax



def prepare_main_input(audioFile, visualFeaturesFile, targetFile, noise, charToIx, noiseSNR, audioParams, videoParams):
    
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


    stftWindow = audioParams["stftWindow"]
    stftWinLen = audioParams["stftWinLen"]
    stftOverlap = audioParams["stftOverlap"]    
    sampFreq, inputAudio = wavfile.read(audioFile)
    if len(inputAudio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
        padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(inputAudio))/2))
        inputAudio = np.pad(inputAudio, padding, "constant")
    inputAudio = inputAudio/np.max(inputAudio)
    if noise is not None:
        pos = np.random.randint(0, len(noise)-len(inputAudio)+1)
        noise = noise[pos:pos+len(inputAudio)]
        noise = noise/np.max(noise)
        gain = 10**(noiseSNR/10)
        noise = noise*np.sqrt(np.sum(inputAudio**2)/(gain*np.sum(noise**2)))
        inputAudio = inputAudio + noise
    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))

    _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq*stftWinLen, 
                                 noverlap=sampFreq*stftOverlap, boundary=None, padded=False)
    audInp = np.abs(stftVals)
    audInp = audInp.T


    vidInp = np.load(visualFeaturesFile)


    if len(audInp)/4 >= len(vidInp):
        inpLen = int(np.ceil(len(audInp)/4))
        padding = (4*inpLen - len(audInp))
        audInp = np.pad(audInp, ((0,padding),(0,0)), "constant")
        padding = (inpLen - len(vidInp))
        vidInp = np.pad(vidInp, ((0,padding),(0,0)), "constant")
    else:
        inpLen = len(vidInp)
        padding = (4*inpLen - len(audInp))
        audInp = np.pad(audInp, ((0,padding),(0,0)), "constant")


    reqInpLen = req_input_length(trgt)
    if inpLen < reqInpLen:
        indices = np.arange(inpLen)
        np.random.shuffle(indices)
        repetitions = int((reqInpLen - inpLen)/inpLen) + 1
        extras = (reqInpLen - inpLen) % inpLen
        newIndices = np.sort(np.concatenate([np.repeat(indices, repetitions), indices[:extras]]))
        audInp = audInp[4*np.repeat(newIndices, 4) + np.tile(np.array([0,1,2,3]), len(newIndices))]
        vidInp = vidInp[newIndices] 

    inpLen = len(vidInp)


    audInp = torch.from_numpy(audInp)
    vidInp = torch.from_numpy(vidInp)
    inp = (audInp,vidInp)
    trgt = torch.from_numpy(trgt)
    inpLen = torch.tensor(inpLen)
    trgtLen = torch.tensor(trgtLen)

    return inp, trgt, inpLen, trgtLen



def prepare_pretrain_input(audioFile, visualFeaturesFile, targetFile, noise, numWords, charToIx, noiseSNR, 
                           audioParams, videoParams):
    
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
        sampFreq, inputAudio = wavfile.read(audioFile)
        vidInp = np.load(visualFeaturesFile) 

    else:
        nWords = [" ".join(words[i:i+numWords]) for i in range(len(words)-numWords+1)]
        nWordLens = np.array([len(nWord)+1 for nWord in nWords]).astype(np.float)
        nWordLens[nWordLens > 256] = -np.inf
        if np.all(nWordLens == -np.inf):
            print("Max target length reached. Exiting")
            exit()
        
        ix = np.random.choice(np.arange(len(nWordLens)), p=softmax(nWordLens))
        trgtNWord = nWords[ix]

        startTime = float(lines[4+ix].split(" ")[1])
        endTime = float(lines[4+ix+numWords-1].split(" ")[2])
        sampFreq, audio = wavfile.read(audioFile)
        inputAudio = audio[int(sampFreq*startTime):int(sampFreq*endTime)]
        vidInp = np.load(visualFeaturesFile)
        vidInp = vidInp[int(np.floor(videoFPS*startTime)):int(np.ceil(videoFPS*endTime))]


    trgt = [charToIx[char] for char in trgtNWord]
    trgt.append(charToIx["<EOS>"])
    trgt = np.array(trgt)
    trgtLen = len(trgt)


    stftWindow = audioParams["stftWindow"]
    stftWinLen = audioParams["stftWinLen"]
    stftOverlap = audioParams["stftOverlap"]
    if len(inputAudio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
        padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(inputAudio))/2))
        inputAudio = np.pad(inputAudio, padding, "constant")
    inputAudio = inputAudio/np.max(inputAudio)
    if noise is not None:
        pos = np.random.randint(0, len(noise)-len(inputAudio)+1)
        noise = noise[pos:pos+len(inputAudio)]
        noise = noise/np.max(noise)
        gain = 10**(noiseSNR/10)
        noise = noise*np.sqrt(np.sum(inputAudio**2)/(gain*np.sum(noise**2)))
        inputAudio = inputAudio + noise
    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))

    _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq*stftWinLen, 
                                 noverlap=sampFreq*stftOverlap, boundary=None, padded=False)
    audInp = np.abs(stftVals)
    audInp = audInp.T


    if len(audInp)/4 >= len(vidInp):
        inpLen = int(np.ceil(len(audInp)/4))
        padding = (4*inpLen - len(audInp))
        audInp = np.pad(audInp, ((0,padding),(0,0)), "constant")
        padding = (inpLen - len(vidInp))
        vidInp = np.pad(vidInp, ((0,padding),(0,0)), "constant")
    else:
        inpLen = len(vidInp)
        padding = (4*inpLen - len(audInp))
        audInp = np.pad(audInp, ((0,padding),(0,0)), "constant")


    reqInpLen = req_input_length(trgt)
    if inpLen < reqInpLen:
        indices = np.arange(inpLen)
        np.random.shuffle(indices)
        repetitions = int((reqInpLen - inpLen)/inpLen) + 1
        extras = (reqInpLen - inpLen) % inpLen
        newIndices = np.sort(np.concatenate([np.repeat(indices, repetitions), indices[:extras]]))
        audInp = audInp[4*np.repeat(newIndices, 4) + np.tile(np.array([0,1,2,3]), len(newIndices))]
        vidInp = vidInp[newIndices] 

    inpLen = len(vidInp)


    audInp = torch.from_numpy(audInp)
    vidInp = torch.from_numpy(vidInp)
    inp = (audInp,vidInp)
    trgt = torch.from_numpy(trgt)
    inpLen = torch.tensor(inpLen)
    trgtLen = torch.tensor(trgtLen)
    
    return inp, trgt, inpLen, trgtLen



def collate_fn(dataBatch):
    inputBatch = (pad_sequence([data[0][0] for data in dataBatch]), 
                  pad_sequence([data[0][1] for data in dataBatch]))
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

