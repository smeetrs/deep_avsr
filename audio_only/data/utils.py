"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.special import softmax



def prepare_main_input(audioFile, targetFile, noise, reqInpLen, charToIx, noiseSNR, audioParams):

    """
    Function to convert the data sample (audio file, target file) in the main dataset into appropriate tensors.
    """

    if targetFile is not None:

        #reading the target from the target file and converting each character to its corresponding index
        with open(targetFile, "r") as f:
            trgt = f.readline().strip()[7:]

        trgt = [charToIx[char] for char in trgt]
        trgt.append(charToIx["<EOS>"])
        trgt = np.array(trgt)
        trgtLen = len(trgt)

        #the target length must be less than or equal to 100 characters (restricted space where our model will work)
        if trgtLen > 100:
            print("Target length more than 100 characters. Exiting")
            exit()


    #STFT feature extraction
    stftWindow = audioParams["stftWindow"]
    stftWinLen = audioParams["stftWinLen"]
    stftOverlap = audioParams["stftOverlap"]
    sampFreq, inputAudio = wavfile.read(audioFile)

    #pad the audio to get atleast 4 STFT vectors
    if len(inputAudio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
        padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(inputAudio))/2))
        inputAudio = np.pad(inputAudio, padding, "constant")
    inputAudio = inputAudio/np.max(np.abs(inputAudio))

    #adding noise to the audio
    if noise is not None:
        pos = np.random.randint(0, len(noise)-len(inputAudio)+1)
        noise = noise[pos:pos+len(inputAudio)]
        noise = noise/np.max(np.abs(noise))
        gain = 10**(noiseSNR/10)
        noise = noise*np.sqrt(np.sum(inputAudio**2)/(gain*np.sum(noise**2)))
        inputAudio = inputAudio + noise

    #normalising the audio to unit power
    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))

    #computing STFT and taking only the magnitude of it
    _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq*stftWinLen, noverlap=sampFreq*stftOverlap,
                                 boundary=None, padded=False)
    inp = np.abs(stftVals)
    inp = inp.T


    #padding zero vectors to make the input length a multiple of 4
    inpLen = int(np.ceil(len(inp)/4))
    leftPadding = int(np.floor((4*inpLen - len(inp))/2))
    rightPadding = int(np.ceil((4*inpLen - len(inp))/2))
    inp = np.pad(inp, ((leftPadding,rightPadding),(0,0)), "constant")


    #checking whether the input length is greater than or equal to the required length
    #if not, extending the input by padding zero vectors
    if inpLen < reqInpLen:
        leftPadding = int(np.floor((reqInpLen - inpLen)/2))
        rightPadding = int(np.ceil((reqInpLen - inpLen)/2))
        inp = np.pad(inp, ((4*leftPadding,4*rightPadding),(0,0)), "constant")

    inpLen = int(len(inp)/4)


    inp = torch.from_numpy(inp)
    inpLen = torch.tensor(inpLen)
    if targetFile is not None:
        trgt = torch.from_numpy(trgt)
        trgtLen = torch.tensor(trgtLen)
    else:
        trgt, trgtLen = None, None

    return inp, trgt, inpLen, trgtLen



def prepare_pretrain_input(audioFile, targetFile, noise, numWords, charToIx, noiseSNR, audioParams):

    """
    Function to convert the data sample (audio file, target file) in the pretrain dataset into appropriate tensors.
    """

    #reading the whole target file and the target
    with open(targetFile, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    trgt = lines[0][7:]
    words = trgt.split(" ")

    #if number of words in target is less than the required number of words, consider the whole target
    if len(words) <= numWords:
        trgtNWord = trgt
        sampFreq, inputAudio = wavfile.read(audioFile)

    else:
        #make a list of all possible sub-sequences with required number of words in the target
        nWords = [" ".join(words[i:i+numWords]) for i in range(len(words)-numWords+1)]
        nWordLens = np.array([len(nWord)+1 for nWord in nWords]).astype(np.float)

        #choose the sub-sequence for target according to a softmax distribution of the lengths
        #this way longer sub-sequences (which are more diverse) are selected more often while
        #the shorter sub-sequences (which appear more frequently) are not entirely missed out
        ix = np.random.choice(np.arange(len(nWordLens)), p=softmax(nWordLens))
        trgtNWord = nWords[ix]

        #reading the start and end times in the video corresponding to the selected sub-sequence
        audioStartTime = float(lines[4+ix].split(" ")[1])
        audioEndTime = float(lines[4+ix+numWords-1].split(" ")[2])
        sampFreq, audio = wavfile.read(audioFile)
        inputAudio = audio[int(sampFreq*audioStartTime):int(sampFreq*audioEndTime)]


    #converting each character in target to its corresponding index
    trgt = [charToIx[char] for char in trgtNWord]
    trgt.append(charToIx["<EOS>"])
    trgt = np.array(trgt)
    trgtLen = len(trgt)


    #STFT feature extraction
    stftWindow = audioParams["stftWindow"]
    stftWinLen = audioParams["stftWinLen"]
    stftOverlap = audioParams["stftOverlap"]

    #pad the audio to get atleast 4 STFT vectors
    if len(inputAudio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
        padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(inputAudio))/2))
        inputAudio = np.pad(inputAudio, padding, "constant")
    inputAudio = inputAudio/np.max(np.abs(inputAudio))

    #adding noise to the audio
    if noise is not None:
        pos = np.random.randint(0, len(noise)-len(inputAudio)+1)
        noise = noise[pos:pos+len(inputAudio)]
        noise = noise/np.max(np.abs(noise))
        gain = 10**(noiseSNR/10)
        noise = noise*np.sqrt(np.sum(inputAudio**2)/(gain*np.sum(noise**2)))
        inputAudio = inputAudio + noise

    #normalising the audio to unit power
    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))

    #computing the STFT and taking only the magnitude of it
    _, _, stftVals = signal.stft(inputAudio, sampFreq, window=stftWindow, nperseg=sampFreq*stftWinLen, noverlap=sampFreq*stftOverlap,
                                 boundary=None, padded=False)
    inp = np.abs(stftVals)
    inp = inp.T


    #padding zero vectors to make the input length a multiple of 4
    inpLen = int(np.ceil(len(inp)/4))
    leftPadding = int(np.floor((4*inpLen - len(inp))/2))
    rightPadding = int(np.ceil((4*inpLen - len(inp))/2))
    inp = np.pad(inp, ((leftPadding,rightPadding),(0,0)), "constant")


    #checking whether the input length is greater than or equal to the required length
    #if not, extending the input by padding zero vectors
    reqInpLen = req_input_length(trgt)
    if inpLen < reqInpLen:
        leftPadding = int(np.floor((reqInpLen - inpLen)/2))
        rightPadding = int(np.ceil((reqInpLen - inpLen)/2))
        inp = np.pad(inp, ((4*leftPadding,4*rightPadding),(0,0)), "constant")

    inpLen = int(len(inp)/4)


    inp = torch.from_numpy(inp)
    inpLen = torch.tensor(inpLen)
    trgt = torch.from_numpy(trgt)
    trgtLen = torch.tensor(trgtLen)

    return inp, trgt, inpLen, trgtLen



def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    inputBatch = pad_sequence([data[0] for data in dataBatch])
    if not any(data[1] is None for data in dataBatch):
        targetBatch = torch.cat([data[1] for data in dataBatch])
    else:
        targetBatch = None

    inputLenBatch = torch.stack([data[2] for data in dataBatch])
    if not any(data[3] is None for data in dataBatch):
        targetLenBatch = torch.stack([data[3] for data in dataBatch])
    else:
        targetLenBatch = None

    return inputBatch, targetBatch, inputLenBatch, targetLenBatch



def req_input_length(trgt):
    """
    Function to calculate the minimum required input length from the target.
    Req. Input Length = No. of unique chars in target + No. of repeats in repeated chars (excluding the first one)
    """
    reqLen = len(trgt)
    lastChar = trgt[0]
    for i in range(1, len(trgt)):
        if trgt[i] != lastChar:
            lastChar = trgt[i]
        else:
            reqLen = reqLen + 1
    return reqLen
