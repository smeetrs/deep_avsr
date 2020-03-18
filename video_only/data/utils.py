import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import cv2 as cv
from scipy.special import softmax



def prepare_main_input(visualFeaturesFile, targetFile, reqInpLen, charToIx, videoParams):
    
    """
    Function to convert the data sample (visual features file, target file) in the main dataset into appropriate tensors.
    """
    
    videoFPS = videoParams["videoFPS"] 

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

    #loading the visual features
    inp = np.load(visualFeaturesFile)

    #checking whether the input length is greater than or equal to the max target length (#characters)
    #if not, extending the input by repeating random feature vectors
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
    
    """
    Function to convert the data sample (visual features file, target file) in the pretrain dataset into appropriate tensors.
    """

    videoFPS = videoParams["videoFPS"]
    
    #reading the whole target file and the target
    with open(targetFile, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    
    trgt = lines[0][7:]
    words = trgt.split(" ")

    #if number of words in target is less than the required number of words, consider the whole target
    if len(words) <= numWords:
        trgtNWord = trgt
        #the target length must be less than 256 characters (pytorch ctc loss function limit when using CUDA)
        if len(trgtNWord)+1 > 256:
            print("PyTorch CTC loss function (CUDA) limit exceeded. Exiting")
            exit()
        #loading the visual features
        inp = np.load(visualFeaturesFile)

    else:
        #make a list of all possible sub-sequences with required number of words in the target
        nWords = [" ".join(words[i:i+numWords]) for i in range(len(words)-numWords+1)]
        nWordLens = np.array([len(nWord)+1 for nWord in nWords]).astype(np.float)
        #the target length must be less than 256 characters (pytorch ctc loss function limit when using CUDA)
        nWordLens[nWordLens > 256] = -np.inf
        if np.all(nWordLens == -np.inf):
            print("PyTorch CTC loss function (CUDA) limit exceeded. Exiting")
            exit()
        
        #choose the sub-sequence for target according to a softmax distribution of the lengths
        #this way longer sub-sequences (which are more diverse) are selected more often while 
        #the shorter sub-sequences (which appear more frequently) are not entirely missed out
        ix = np.random.choice(np.arange(len(nWordLens)), p=softmax(nWordLens))
        trgtNWord = nWords[ix]

        #reading the start and end times in the video corresponding to the selected sub-sequence
        videoStartTime = float(lines[4+ix].split(" ")[1])
        videoEndTime = float(lines[4+ix+numWords-1].split(" ")[2])
        #loading the visual features
        inp = np.load(visualFeaturesFile)
        inp = inp[int(np.floor(videoFPS*videoStartTime)):int(np.ceil(videoFPS*videoEndTime))]


    #converting each character in target to its corresponding index
    trgt = [charToIx[char] for char in trgtNWord]
    trgt.append(charToIx["<EOS>"])
    trgt = np.array(trgt)
    trgtLen = len(trgt)
    
    #checking whether the input length is greater than or equal to the target length (#characters)
    #if not, extending the input by repeating random feature vectors
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
    """
    Collate function definition used in Dataloaders. 
    """
    inputBatch = pad_sequence([data[0] for data in dataBatch])
    targetBatch = torch.cat([data[1] for data in dataBatch])
    inputLenBatch = torch.stack([data[2] for data in dataBatch])
    targetLenBatch = torch.stack([data[3] for data in dataBatch])
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

