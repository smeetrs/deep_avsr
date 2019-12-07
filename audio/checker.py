import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

from config import args
from models.lrs2_char_lm import LRS2CharLM
from models.audio_net import AudioNet
from data.utils import req_input_length, collate_fn
from data.lrs2_dataset import LRS2Pretrain, LRS2Main
from utils.decoders import ctc_greedy_decode, ctc_search_decode
from utils.metrics import compute_cer, compute_wer


def req_input_length_checker():
    strings = ["WORKS FOR DOODEE OOTY ASSAM", "       ", "NOOOOOOOOOO", "IT'S THAT SIMPLE"]
    for n in range(len(strings)):
        trgt = list()
        for i in range(len(strings[n])):
            char = strings[n][i]
            ix = args["CHAR_TO_INDEX"][char]
            trgt.append(ix)
        print(req_input_length(trgt))
    return


def collate_fn_checker():
    dataBatch = list()
    inpLens = [40, 35, 28, 42]
    trgtLens = [4, 6, 7, 10]
    for i in range(len(inpLens)):
        inp = torch.from_numpy(np.random.rand(inpLens[i], args["AUDIO_FEATURE_SIZE"]))
        trgt = torch.from_numpy(np.random.randint(0,args["NUM_CLASSES"],trgtLens[i]))
        inpLen = torch.tensor(inpLens[i])
        trgtLen = torch.tensor(trgtLens[i])
        data = (inp, trgt, inpLen, trgtLen)
        dataBatch.append(data)
    inputBatch, targetBatch, inputLenBatch, targetLenBatch = collate_fn(dataBatch)
    print(inputBatch.size(), targetBatch.size(), inputLenBatch.size(), targetLenBatch.size())
    return


def lrs2pretrain_checker():
    pretrainData = LRS2Pretrain(datadir=args["DATA_DIRECTORY"], numWords=args["PRETRAIN_NUM_WORDS"], 
                                charToIx=args["CHAR_TO_INDEX"], stepSize=args["STEP_SIZE"], 
                                stftParams={"window":args["STFT_WINDOW"], 
                                            "winLen":args["STFT_WIN_LENGTH"], 
                                            "overlap":args["STFT_OVERLAP"]})
    numSamples = len(pretrainData)
    index = np.random.randint(0, numSamples)
    inp, trgt, inpLen, trgtLen = pretrainData[index]
    print(inp.size(), trgt.size(), inpLen.size(), trgtLen.size())
    return


def lrs2main_checker():
    trainData = LRS2Main(dataset="train", datadir=args["DATA_DIRECTORY"], charToIx=args["CHAR_TO_INDEX"], 
                         stepSize=args["STEP_SIZE"], stftParams={"window":args["STFT_WINDOW"], 
                                                                 "winLen":args["STFT_WIN_LENGTH"], 
                                                                 "overlap":args["STFT_OVERLAP"]})
    numSamples = len(trainData)
    index = np.random.randint(0, numSamples)
    inp, trgt, inpLen, trgtLen = trainData[index]
    print(inp.size(), trgt.size(), inpLen.size(), trgtLen.size())
    return


def lrs2main_max_inplen_checker():
    maxInpLen = 0
    for root, dirs, files in os.walk(args["DATA_DIRECTORY"] + "/main"):
        for file in files:
            if file.endswith(".mp4"):
                audioFile = os.path.join(root, file[:-4]) + ".wav"
                targetFile = os.path.join(root, file[:-4]) + ".txt"
                with open(targetFile, "r") as f:
                    trgt = f.readline().strip()[7:]
                sampFreq, audio = wavfile.read(audioFile)
                inpLen = (len(audio) - 640)//160 + 1
                reqLen = req_input_length(trgt)
                if reqLen*4 > inpLen:
                    inpLen = reqLen*4
                if inpLen > maxInpLen:
                    maxInpLen = inpLen
    print(maxInpLen)
    return


def trgtlen_distribution_checker():
    for dataset in ["pretrain","main"]:
        distribution = np.zeros(2500, dtype=np.int)
        for root, dirs, files in os.walk(args["DATA_DIRECTORY"] + "/" + dataset):
            for file in files:
                if file.endswith(".mp4"):
                    targetFile = os.path.join(root, file[:-4]) + ".txt"
                    with open(targetFile, "r") as f:
                        trgt = f.readline().strip()[7:]
                        trgtLen = len(trgt)
                        distribution[trgtLen] = distribution[trgtLen] + 1

        for i in range(len(distribution)):
            if distribution[i] != 0:
                print("Dataset: %s, Min Target Length = %d" %(dataset, i))
                break

        for i in range(len(distribution)-1, -1, -1):
            if distribution[i] != 0:
                print("Dataset: %s, Max Target Length = %d" %(dataset, i))
                break

        plt.figure()
        plt.title("{} dataset target length distribution".format(dataset))
        plt.xlabel("Target Lengths")
        plt.ylabel("Counts")
        plt.bar(np.arange(2500), distribution)
        plt.savefig(args["DATA_DIRECTORY"] + "/" + dataset + ".png")
        plt.close()
    return


def word_length_distribution_checker():
    distribution = np.zeros(35, dtype=np.int)
    for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
        for file in files:
            if file.endswith(".mp4"):
                targetFile = os.path.join(root, file[:-4]) + ".txt"
                with open(targetFile, "r") as f:
                    trgt = f.readline().strip()[7:]
                    words = trgt.split(" ")
                    wordLens = np.array([len(word) for word in words])
                    distribution = distribution + np.histogram(wordLens, bins=np.arange(36))[0]

    for i in range(len(distribution)):
        if distribution[i] != 0:
            print("Min Word Length = %d" %(i))
            break

    for i in range(len(distribution)-1, -1, -1):
        if distribution[i] != 0:
            print("Max Word Length = %d" %(i))
            break

    plt.figure()
    plt.title("Word length distribution")
    plt.xlabel("Word Lengths")
    plt.ylabel("Counts")
    plt.bar(np.arange(35), distribution)
    plt.savefig(args["DATA_DIRECTORY"] + "/word.png")
    plt.close()
    return


def lrs2pretrain_max_inplen_checker():
    maxInpLen = 0
    numWords = args["PRETRAIN_NUM_WORDS"]
    for root, dirs, files in os.walk(args["DATA_DIRECTORY"] + "/pretrain"):
        for file in files:
            if file.endswith(".mp4"):

                audioFile = os.path.join(root, file[:-4]) + ".wav"
                targetFile = os.path.join(root, file[:-4]) + ".txt"
                with open(targetFile, "r") as f:
                    lines = f.readlines()
                lines = [line.strip() for line in lines]
                trgt = lines[0][7:]
                words = trgt.split(" ")

                if len(words) <= numWords:
                    if len(trgt) > 256:
                        print("Max target length reached. Exiting")
                        exit()
                    sampFreq, audio = wavfile.read(audioFile)
                    inpLen = (len(audio) - 640)//160 + 1
                    reqLen = req_input_length(trgt)
                    if reqLen*4 > inpLen:
                        inpLen = reqLen*4
                    if inpLen > maxInpLen:
                        maxInpLen = inpLen

                else:
                    nWords = np.array([" ".join(words[i:i+numWords]) for i in range(len(words) - numWords + 1)])
                    nWordLens = np.array([len(nWord) for nWord in nWords]).astype(np.float)
                    nWordLens[nWordLens > 256] = -np.inf
                    if np.all(nWordLens == -np.inf):
                        print("Max target length reached. Exiting")
                        exit()     

                    nWords = nWords[nWordLens > 0]       
                    for ix in range(len(nWords)):
                        trgtNWord = nWords[ix]
                        audioStartTime = float(lines[4+ix].split(" ")[1])
                        audioEndTime = float(lines[4+ix+numWords-1].split(" ")[2])
                        sampFreq, audio = wavfile.read(audioFile)
                        inputAudio = audio[int(sampFreq*audioStartTime):int(sampFreq*audioEndTime)]
                        inpLen = (len(inputAudio) - 640)//160 + 1
                        if len(inputAudio) < (640 + 3*160):
                            inpLen = 4
                        reqLen = req_input_length(trgt) + 1
                        if reqLen*4 > inpLen:
                            inpLen = reqLen*4
                        if inpLen > maxInpLen:
                            maxInpLen = inpLen                            
    print(maxInpLen)
    return


def ctc_greedy_decode_checker():
    outputs = ["TTTEEEST-IINNNN-G-   CC-TTCCC- -DEEE-CO-DD---E       -FUU-NCCC--TAA-B-FA--E", 
               "ONNE SSSTEEEP    ISSS  OOOOVVEERA- FDDA-S A FD-AASDF - AD-AFA DF-ADF SF-ADF", 
               "EVERYTHING ALRIGHT CHECK DONE SH-SG-GAD-G HS- RA-R H J- J-AM GA-AM GA-GA-AD", 
               "SSEEEE-E--  -EEE-VE-NNN  ---DDDOOOO-ODDE-E   --O-OOOOTTTY AAAASS-SSAAM WORK",
               "---------------------------------------------------------------------------"]
    inpLens = [64, 32, 29, 75, 56]

    outputProbs = 0.01*torch.zeros((max(inpLens), len(inpLens), args["NUM_CLASSES"]))
    inpLens = torch.tensor(inpLens)
    for n in range(len(outputs)):
        for i in range(len(outputs[n])):
            char = outputs[n][i]
            if char == "-":
                ix = 0
            else:
                ix = args["CHAR_TO_INDEX"][char]
            outputProbs[i,n,ix] = 0.61
    outputLogProbs = torch.log_softmax(outputProbs, dim=2)

    predictions, predictionLens = ctc_greedy_decode(outputLogProbs, inpLens)
    predictions = [args["INDEX_TO_CHAR"][ix] for ix in predictions.tolist()]
    predictedSequences = list()
    s = 0
    for ln in predictionLens.tolist():
        predictedSequences.append("".join(predictions[s:s+ln]))
        s = s + ln
    print(predictedSequences)
    return


def ctc_search_decode_checker():
    outputs = ["TTTEEEST-IINNNN-G-   CC-TTCCC- -DEEE-CO-DD---E       -FUU-NCCC--TAA-B-FA--E", 
               "ONNE SSSTEEEP    ISSS  OOOOVVEERA- FDDA-S A FD-AASDF - AD-AFA DF-ADF SF-ADF", 
               "EVERYTHING ALRIGHT CHECK DONE SH-SG-GAD-G HS- RA-R H J- J-AM GA-AM GA-GA-AD", 
               "SSEEEE-E--  -EEE-VE-NNN  ---DDDOOOO-ODDE-E   --O-OOOOTTTY AAAASS-SSAAM WORK",
               "---------------------------------------------------------------------------"]
    inpLens = [64, 32, 29, 75, 56]

    outputProbs = 0.01*torch.zeros((len(outputs[0]), len(inpLens), args["NUM_CLASSES"]))
    inpLens = torch.tensor(inpLens)
    for n in range(len(outputs)):
        for i in range(len(outputs[n])):
            char = outputs[n][i]
            if char == "-":
                ix = 0
            else:
                ix = args["CHAR_TO_INDEX"][char]
            outputProbs[i,n,ix] = 0.61
    outputLogProbs = torch.log(outputProbs)

    beamSearchParams = {"beamWidth":args["BEAM_WIDTH"], "alpha":args["LM_WEIGHT_ALPHA"], "beta":args["LENGTH_PENALTY_BETA"]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = LRS2CharLM().to(device)
    lm.load_state_dict(torch.load(args["TRAINED_LM_FILE"]))
    lm.to(device)

    predictions, predictionLens = ctc_search_decode(outputLogProbs, inpLens, 
                                                    beamSearchParams, spaceIx=args["CHAR_TO_INDEX"][" "], lm=lm)
    predictions = [args["INDEX_TO_CHAR"][ix] for ix in predictions.tolist()]
    predictedSequences = list()
    s = 0
    for ln in predictionLens.tolist():
        predictedSequences.append("".join(predictions[s:s+ln]))
        s = s + ln
    print(predictedSequences)
    return


def lrs2charlm_checker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LRS2CharLM().to(device)
    model.load_state_dict(torch.load(args["TRAINED_LM_FILE"]))
    model.to(device)
    model.eval()

    inp = torch.tensor(args["CHAR_TO_INDEX"][" "]-1)
    initStateBatch = None
    string = list()
    for i in range(100):
        inputBatch = inp.view(1,1)
        inputBatch = inputBatch.to(device)
        with torch.no_grad():
            outputBatch, finalStateBatch = model(inputBatch, initStateBatch)
        
        outputBatch = torch.exp(outputBatch)
        out = outputBatch.view(outputBatch.size(2))
        probs = out.tolist()
        ix = np.random.choice(np.arange(len(probs)), p=probs/np.sum(probs))
        char = args["INDEX_TO_CHAR"][ix+1]
        string.append(char)

        inp = torch.tensor(ix)
        initStateBatch = finalStateBatch

    print("".join(string))
    return


def audionet_checker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioNet(dModel=args["TX_NUM_FEATURES"], nHeads=args["TX_ATTENTION_HEADS"], 
                     numLayers=args["TX_NUM_LAYERS"], peMaxLen=args["PE_MAX_LENGTH"], 
                     inSize=args["AUDIO_FEATURE_SIZE"], fcHiddenSize=args["TX_FEEDFORWARD_DIM"], 
                     dropout=args["TX_DROPOUT"], numClasses=args["NUM_CLASSES"])
    model.to(device)
    T, N, C = 42, args["BATCH_SIZE"], args["AUDIO_FEATURE_SIZE"]
    inputBatch = torch.rand(T, N, C).to(device)
    outputBatch = model(inputBatch)
    print(outputBatch.size())
    return


def compute_wer_checker():
    preds = [" SOMETH'NG  '  NEE DS TO BE D'NE   ABOUT IT", "FUNCTION CHECKING INITIATED", "    '   ", ""]
    trgts = ["SOMETHING NEEDS TO BE DONE ABOUT IT", "FUNCTION CHECKING INITIATED", "SOME ARBIT STRING", "ARBIT STRING"]
    predLens = [43, 27, 8, 0]
    trgtLens = [35, 27, 17, 12]

    predIxs = list()
    for n in range(len(preds)):
        predIx = list()
        for i in range(len(preds[n])):
            char = preds[n][i]
            ix = args["CHAR_TO_INDEX"][char]
            predIx.append(ix)
        predIxs.extend(predIx)

    trgtIxs = list()
    for n in range(len(trgts)):
        trgtIx = list()
        for i in range(len(trgts[n])):
            char = trgts[n][i]
            ix = args["CHAR_TO_INDEX"][char]
            trgtIx.append(ix)
        trgtIxs.extend(trgtIx)

    predictionBatch = torch.tensor(predIxs)
    targetBatch = torch.tensor(trgtIxs)
    predictionLenBatch = torch.tensor(predLens)
    targetLenBatch = torch.tensor(trgtLens)

    print(compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, spaceIx=args["CHAR_TO_INDEX"][" "]))
    return


def compute_cer_checker():
    preds = ["SOMETIN'  ' NEDSS", "   ALRIT ", "CHEK DON", "EXACT SAME"]
    trgts = ["SOMETHING NEEDS", "ALRIGHT", "CHECK DONE", "EXACT SAME"]
    predLens = [17, 9, 8, 10]
    trgtLens = [15, 7, 10, 10]

    predIxs = list()
    for n in range(len(preds)):
        predIx = list()
        for i in range(len(preds[n])):
            char = preds[n][i]
            ix = args["CHAR_TO_INDEX"][char]
            predIx.append(ix)
        predIxs.extend(predIx)

    trgtIxs = list()
    for n in range(len(trgts)):
        trgtIx = list()
        for i in range(len(trgts[n])):
            char = trgts[n][i]
            ix = args["CHAR_TO_INDEX"][char]
            trgtIx.append(ix)
        trgtIxs.extend(trgtIx)

    predictionBatch = torch.tensor(predIxs)
    targetBatch = torch.tensor(trgtIxs)
    predictionLenBatch = torch.tensor(predLens)
    targetLenBatch = torch.tensor(trgtLens)

    print(compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch))
    return


if __name__ == '__main__':
    #call the required function checker
    compute_cer_checker()
    compute_wer_checker()
