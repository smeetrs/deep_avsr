"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/smeetrs/deep_avsr
"""

import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

from config import args
from models.lrs2_char_lm import LRS2CharLM
from models.video_net import VideoNet
from models.visual_frontend import VisualFrontend
from data.utils import req_input_length, collate_fn
from data.lrs2_dataset import LRS2Pretrain, LRS2Main
from utils.decoders import ctc_greedy_decode, ctc_search_decode
from utils.metrics import compute_cer, compute_wer
from utils.preprocessing import preprocess_sample



def req_input_length_checker():
    strings = ["WORKS FOR DOODEE OOTY ASSAM~", "       ~", "NOOOOOOOOOO~", "IT'S THAT SIMPLE~"]
    for n in range(len(strings)):
        trgt = list()
        for i in range(len(strings[n])):
            char = strings[n][i]
            if char == "~":
                ix = args["CHAR_TO_INDEX"]["<EOS>"]
            else:
                ix = args["CHAR_TO_INDEX"][char]
            trgt.append(ix)
        print(req_input_length(trgt))
    return


def collate_fn_checker():
    dataBatch = list()
    inpLens = [10, 8, 7, 10]
    trgtLens = [4, 6, 7, 10]
    for i in range(len(inpLens)):
        inp = torch.from_numpy(np.random.rand(inpLens[i], 1, args["ROI_SIZE"], args["ROI_SIZE"]))
        trgt = torch.from_numpy(np.random.randint(0, args["NUM_CLASSES"], trgtLens[i]))
        inpLen = torch.tensor(inpLens[i])
        trgtLen = torch.tensor(trgtLens[i])
        data = (inp, trgt, inpLen, trgtLen)
        dataBatch.append(data)
    inputBatch, targetBatch, inputLenBatch, targetLenBatch = collate_fn(dataBatch)
    print(inputBatch.shape, targetBatch.shape, inputLenBatch.shape, targetLenBatch.shape)
    return


def lrs2pretrain_checker():
    videoParams = {"videoFPS":args["VIDEO_FPS"]}
    pretrainData = LRS2Pretrain(args["DATA_DIRECTORY"], args["PRETRAIN_NUM_WORDS"], args["CHAR_TO_INDEX"], args["STEP_SIZE"], videoParams)
    numSamples = len(pretrainData)
    index = np.random.randint(0, numSamples)
    inp, trgt, inpLen, trgtLen = pretrainData[index]
    print(inp.shape, trgt.shape, inpLen.shape, trgtLen.shape)
    return


def lrs2main_checker():
    videoParams = {"videoFPS":args["VIDEO_FPS"]}
    trainData = LRS2Main("train", args["DATA_DIRECTORY"], args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                         videoParams)
    numSamples = len(trainData)
    index = np.random.randint(0, numSamples)
    inp, trgt, inpLen, trgtLen = trainData[index]
    print(inp.shape, trgt.shape, inpLen.shape, trgtLen.shape)
    return


def lrs2main_max_inplen_checker():
    maxInpLen = 0
    for root, dirs, files in os.walk(args["DATA_DIRECTORY"] + "/main"):
        for file in files:
            if file.endswith(".mp4"):
                visualFeaturesFile = os.path.join(root, file[:-4]) + ".npy"
                targetFile = os.path.join(root, file[:-4]) + ".txt"
                with open(targetFile, "r") as f:
                    trgt = f.readline().strip()[7:]
                visualFeatures = np.load(visualFeaturesFile)
                inpLen = len(visualFeatures)
                reqLen = req_input_length(trgt)+1
                if reqLen > inpLen:
                    inpLen = reqLen
                if inpLen > maxInpLen:
                    maxInpLen = inpLen
    print(maxInpLen)
    return


def trgtlen_distribution_checker():
    for dataset in ["pretrain","main"]:
        if dataset == "pretrain":
            distribution = np.zeros(2500, dtype=np.int)
        else:
            distribution = np.zeros(150, dtype=np.int)
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
        if dataset == "pretrain":
            plt.bar(np.arange(2500), distribution)
        else:
            plt.bar(np.arange(150), distribution)
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

                visualFeaturesFile = os.path.join(root, file[:-4]) + ".npy"
                targetFile = os.path.join(root, file[:-4]) + ".txt"
                with open(targetFile, "r") as f:
                    lines = f.readlines()
                lines = [line.strip() for line in lines]
                trgt = lines[0][7:]
                words = trgt.split(" ")

                if len(words) <= numWords:
                    if len(trgt)+1 > 256:
                        print("Max target length reached. Exiting")
                        exit()
                    visualFeatures = np.load(visualFeaturesFile)
                    inpLen = len(visualFeatures)
                    reqLen = req_input_length(trgt)+1
                    if reqLen > inpLen:
                        inpLen = reqLen
                    if inpLen > maxInpLen:
                        maxInpLen = inpLen

                else:
                    nWords = np.array([" ".join(words[i:i+numWords]) for i in range(len(words) - numWords + 1)])
                    nWordLens = np.array([len(nWord)+1 for nWord in nWords]).astype(np.float)
                    nWordLens[nWordLens > 256] = -np.inf
                    if np.all(nWordLens == -np.inf):
                        print("Max target length reached. Exiting")
                        exit()

                    nWords = nWords[nWordLens > 0]
                    for ix in range(len(nWords)):
                        trgt = nWords[ix]
                        videoStartTime = float(lines[4+ix].split(" ")[1])
                        videoEndTime = float(lines[4+ix+numWords-1].split(" ")[2])
                        inpLen = int(np.ceil(args["VIDEO_FPS"]*videoEndTime) - np.floor(args["VIDEO_FPS"]*videoStartTime))
                        reqLen = req_input_length(trgt)+1
                        if reqLen > inpLen:
                            inpLen = reqLen
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

    outputProbs = 0.01*torch.ones((max(inpLens), len(inpLens), args["NUM_CLASSES"]))
    inpLens = torch.tensor(inpLens)
    for n in range(len(outputs)):
        for i in range(len(outputs[n])):
            char = outputs[n][i]
            if char == "-":
                ix = 0
            else:
                ix = args["CHAR_TO_INDEX"][char]
            outputProbs[i,n,ix] = 1.5
    outputLogProbs = torch.log_softmax(outputProbs, dim=2)

    predictions, predictionLens = ctc_greedy_decode(outputLogProbs, inpLens, args["CHAR_TO_INDEX"]["<EOS>"])
    predictions = [args["INDEX_TO_CHAR"][ix] for ix in predictions.tolist() if ix != args["CHAR_TO_INDEX"]["<EOS>"]]
    predictedSequences = list()
    s = 0
    for ln in predictionLens.tolist():
        predictedSequences.append("".join(predictions[s:s+ln-1]))
        s = s + ln - 1
    print(predictedSequences)
    return


def ctc_search_decode_checker():
    outputs = ["TTTEEEST-IINNNN-G-   CC-TTCCC- -DEEE-CO-DD---E       -FUU-NCCC--TAA-B-FA--E",
               "ONNE SSSTEEEP    ISSS  OOOOVVEERA- FDDA-S A FD-AASDF - AD-AFA DF-ADF SF-ADF",
               "EVERYTHING ALRIGHT CHECK DONE SH-SG-GAD-G HS- RA-R H J- J-AM GA-AM GA-GA-AD",
               "SSEEEE-E--  -EEE-VE-NNN  ---DDDOOOO-ODDE-E   --O-OOOOTTTY AAAASS-SSAAM WORK",
               "---------------------------------------------------------------------------"]
    inpLens = [64, 32, 29, 75, 56]

    outputProbs = 0.01*torch.ones((len(outputs[0]), len(inpLens), args["NUM_CLASSES"]))
    inpLens = torch.tensor(inpLens)
    for n in range(len(outputs)):
        for i in range(len(outputs[n])):
            char = outputs[n][i]
            if char == "-":
                ix = 0
            else:
                ix = args["CHAR_TO_INDEX"][char]
            outputProbs[i,n,ix] = 1.5
    outputLogProbs = torch.log(outputProbs)

    beamSearchParams = {"beamWidth":args["BEAM_WIDTH"], "alpha":args["LM_WEIGHT_ALPHA"], "beta":args["LENGTH_PENALTY_BETA"],
                        "threshProb":args["THRESH_PROBABILITY"]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = LRS2CharLM()
    lm.load_state_dict(torch.load(args["TRAINED_LM_FILE"], map_location=device))
    lm.to(device)
    if not args["USE_LM"]:
        lm = None

    predictions, predictionLens = ctc_search_decode(outputLogProbs, inpLens, beamSearchParams, args["CHAR_TO_INDEX"][" "],
                                                    args["CHAR_TO_INDEX"]["<EOS>"], lm)
    predictions = [args["INDEX_TO_CHAR"][ix] for ix in predictions.tolist() if ix != args["CHAR_TO_INDEX"]["<EOS>"]]
    predictedSequences = list()
    s = 0
    for ln in predictionLens.tolist():
        predictedSequences.append("".join(predictions[s:s+ln-1]))
        s = s + ln - 1
    print(predictedSequences)
    return


def lrs2charlm_checker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LRS2CharLM()
    model.load_state_dict(torch.load(args["TRAINED_LM_FILE"], map_location=device))
    model.to(device)

    inp = torch.tensor(args["CHAR_TO_INDEX"][" "]-1)
    initStateBatch = None
    string = list()
    for i in range(100):
        inputBatch = inp.reshape(1,1)
        inputBatch = inputBatch.to(device)
        model.eval()
        with torch.no_grad():
            outputBatch, finalStateBatch = model(inputBatch, initStateBatch)

        outputBatch = torch.exp(outputBatch)
        out = outputBatch.squeeze()
        probs = out.tolist()
        ix = np.random.choice(np.arange(len(probs)), p=probs/np.sum(probs))
        char = args["INDEX_TO_CHAR"][ix+1]
        string.append(char)

        inp = torch.tensor(ix)
        initStateBatch = finalStateBatch

    print("".join(string))
    return


def videonet_checker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoNet(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
                     args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"])
    model.to(device)
    T, N, C = 10, args["BATCH_SIZE"], args["TX_NUM_FEATURES"]
    inputBatch = torch.rand(T, N, C).to(device)
    model.eval()
    with torch.no_grad():
        outputBatch = model(inputBatch)
    print(outputBatch.shape)
    return


def visualfrontend_checker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisualFrontend().to(device)
    model.to(device)
    model.eval()
    T, N, C, H, W = 10, args["BATCH_SIZE"], 1, args["ROI_SIZE"], args["ROI_SIZE"]
    inputBatch = torch.rand(T, N, C, H, W).to(device)
    with torch.no_grad():
        outputBatch = model(inputBatch)
    print(outputBatch.shape)
    return


def compute_wer_checker():
    preds = [" SOMETH'NG  '  NEE DS TO BE D'NE   ABOUT IT~", "FUNCTION CHECKING INITIATED~", "    '   ~", "~"]
    trgts = ["SOMETHING NEEDS TO BE DONE ABOUT IT~", "FUNCTION CHECKING INITIATED~", "SOME ARBIT STRING~", "ARBIT STRING~"]
    predLens = [44, 28, 9, 1]
    trgtLens = [36, 28, 18, 13]

    predIxs = list()
    for n in range(len(preds)):
        predIx = list()
        for i in range(len(preds[n])):
            char = preds[n][i]
            if char == "~":
                ix = args["CHAR_TO_INDEX"]["<EOS>"]
            else:
                ix = args["CHAR_TO_INDEX"][char]
            predIx.append(ix)
        predIxs.extend(predIx)

    trgtIxs = list()
    for n in range(len(trgts)):
        trgtIx = list()
        for i in range(len(trgts[n])):
            char = trgts[n][i]
            if char == "~":
                ix = args["CHAR_TO_INDEX"]["<EOS>"]
            else:
                ix = args["CHAR_TO_INDEX"][char]
            trgtIx.append(ix)
        trgtIxs.extend(trgtIx)

    predictionBatch = torch.tensor(predIxs)
    targetBatch = torch.tensor(trgtIxs)
    predictionLenBatch = torch.tensor(predLens)
    targetLenBatch = torch.tensor(trgtLens)

    print(compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, args["CHAR_TO_INDEX"][" "]))
    return


def compute_cer_checker():
    preds = ["SOMETIN'  ' NEDSS~", "   ALRIT ~", "CHEK DON~", "EXACT SAME~"]
    trgts = ["SOMETHING NEEDS~", "ALRIGHT~", "CHECK DONE~", "EXACT SAME~"]
    predLens = [18, 10, 9, 11]
    trgtLens = [16, 8, 11, 11]

    predIxs = list()
    for n in range(len(preds)):
        predIx = list()
        for i in range(len(preds[n])):
            char = preds[n][i]
            if char == "~":
                ix = args["CHAR_TO_INDEX"]["<EOS>"]
            else:
                ix = args["CHAR_TO_INDEX"][char]
            predIx.append(ix)
        predIxs.extend(predIx)

    trgtIxs = list()
    for n in range(len(trgts)):
        trgtIx = list()
        for i in range(len(trgts[n])):
            char = trgts[n][i]
            if char == "~":
                ix = args["CHAR_TO_INDEX"]["<EOS>"]
            else:
                ix = args["CHAR_TO_INDEX"][char]
            trgtIx.append(ix)
        trgtIxs.extend(trgtIx)

    predictionBatch = torch.tensor(predIxs)
    targetBatch = torch.tensor(trgtIxs)
    predictionLenBatch = torch.tensor(predLens)
    targetLenBatch = torch.tensor(trgtLens)

    print(compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch))
    return


def preprocess_sample_checker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vf = VisualFrontend()
    vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"], map_location=device))
    vf.to(device)
    file = args["CODE_DIRECTORY"] + "/demo/00001"
    params = {"roiSize":args["ROI_SIZE"], "normMean":args["NORMALIZATION_MEAN"], "normStd":args["NORMALIZATION_STD"], "vf":vf}
    preprocess_sample(file, params)
    return


def word_count_distribution_checker():
    for dataset in ["pretrain","main"]:
        if dataset == "pretrain":
            distribution = np.zeros(500, dtype=np.int)
        else:
            distribution = np.zeros(30, dtype=np.int)
        for root, dirs, files in os.walk(args["DATA_DIRECTORY"] + "/" + dataset):
            for file in files:
                if file.endswith(".mp4"):
                    targetFile = os.path.join(root, file[:-4]) + ".txt"
                    with open(targetFile, "r") as f:
                        trgt = f.readline().strip()[7:]
                        words = trgt.split(" ")
                        numWords = len(words)
                        distribution[numWords] = distribution[numWords] + 1

        for i in range(len(distribution)):
            if distribution[i] != 0:
                print("Dataset: %s, Min Word Count = %d" %(dataset, i))
                break

        for i in range(len(distribution)-1, -1, -1):
            if distribution[i] != 0:
                print("Dataset: %s, Max Word Count = %d" %(dataset, i))
                break

        plt.figure()
        plt.title("{} dataset word count distribution".format(dataset))
        plt.xlabel("Number of Words in a Target")
        plt.ylabel("Counts")
        if dataset == "pretrain":
            plt.bar(np.arange(500), distribution)
        else:
            plt.bar(np.arange(30), distribution)
        plt.savefig(args["DATA_DIRECTORY"] + "/" + dataset + "_wd.png")
        plt.close()
    return


if __name__ == "__main__":
    #call the required function checker
    #delete the function calls after checking to avoid pushing everytime to github
    exit()
