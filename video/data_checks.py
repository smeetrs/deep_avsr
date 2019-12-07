import argparse
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile


parser = argparse.ArgumentParser()
parser.add_argument("--datadir", default="./data", help="Path to data")
parser.add_argument("--check", required=True, help="Check to be performed")
args = parser.parse_args()


def required_input_length(string):
    reqLen = len(string)
    lastChar = string[0]
    for i in range(1, len(string)):
        if string[i] != lastChar:
            lastChar = string[i]
        else:
            reqLen = reqLen + 1
    return reqLen




def input_ge_target():
    
    videoFilesList = list()
    for root, dirs, files in os.walk(args.datadir):
        for file in files:
            if file.endswith(".mp4"):
                videoFilesList.append(os.path.join(root, file))


    print("Files which don't satisfy the criteria:")
    for file in videoFilesList:
        
        videofile = file
        textfile = file[:-4] + ".txt"
        
        cap = cv.VideoCapture(videofile)
        inputLen = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        with open(textfile, "r") as f:
            target = f.readline().strip()[7:].replace("'", "")
            targetLen = len(target) 
        
        if (inputLen < targetLen):
            print("%s Inlen=%d Tarlen=%d" %(file[:-4], inputLen, targetLen))
    
    return



def target_len_distribution(data):
    
    if data == "pretrain":
        with open(args.datadir + "/pretrain.txt", "r") as f:
            filesList = f.readlines()
        filesList = [args.datadir + "/pretrain/" + file.strip() + ".txt" for file in filesList]

    else:
        with open(args.datadir + "/" + data + ".txt", "r") as f:
            filesList = f.readlines()
        filesList = [args.datadir + "/main/" + file.strip().split(" ")[0] + ".txt" for file in filesList]

    distribution = np.zeros(2500, dtype=np.int)
    for file in filesList:
        with open(file, "r") as f:
            target = f.readline().strip()[7:].replace("'", "")
            targetLen = len(target)
            distribution[targetLen] = distribution[targetLen] + 1

    for i in range(len(distribution)):
        if distribution[i] != 0:
            print("Min Target Length = %d" %(i))
            break

    for i in range(len(distribution)-1, -1, -1):
        if distribution[i] != 0:
            print("Max Target Length = %d" %(i))
            break

    plt.figure()
    plt.title("{} dataset target length distribution".format(data))
    plt.xlabel("Target Lengths")
    plt.ylabel("Counts")
    plt.bar(np.arange(2500), distribution)
    plt.show()

    return


def word_len_distribution():

    filesList = list()
    for root, dirs, files in os.walk(args.datadir):
        for file in files:
            if file.endswith(".mp4"):
                filesList.append(os.path.join(root, file[:-4] + ".txt"))

    distribution = np.zeros(35, dtype=np.int)
    for file in filesList:
        with open(file, "r") as f:
            target = f.readline().strip()[7:].replace("'", "")
            words = target.split(" ")
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
    plt.show()

    return



def max_fcount_chars_ratio():
    
    filesList = list()
    for root, dirs, files in os.walk(args.datadir):
        for file in files:
            if file.endswith(".mp4"):
                filesList.append(os.path.join(root, file[:-4]))

    maxRatio = 0
    for file in filesList:
        videofile = file + ".mp4"
        targetFile = file + ".txt"

        cap = cv.VideoCapture(videofile)
        frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        with open(targetFile, "r") as f:
            target = f.readline().strip()[7:].replace("'", "")
            targetLen = len(target)+1

        ratio = frameCount/targetLen
        if ratio > maxRatio:
            maxRatio = ratio

    print(maxRatio)

    return



def pretrain_max_inplen(numWords):
    
    with open(args.datadir + "/pretrain.txt", "r") as f:
        filesList = f.readlines()
    filesList = [args.datadir + "/pretrain/" + file.strip() for file in filesList]

    maxTarget = None
    maxInpLen = 0

    for file in filesList:
        audioFile = file + ".wav"
        targetFile = file + ".txt"

        with open(targetFile, "r") as f:
            lines = f.readlines()
        lines = [line.strip().replace("'","") for line in lines]

        target = lines[0][7:]
        words = target.split(" ")

        if len(words) <= numWords:

            if len(target)+1 > 256:
                print("Max target length reached. Exiting")
                exit()
            sampFreq, audio = wavfile.read(audioFile)
            inpLen = (len(audio) - 640)//160 + 1
            if inpLen > maxInpLen:
                maxInpLen = inpLen
                maxTarget = target

        else:

            nWords = np.array([" ".join(words[i:i+numWords]) for i in range(len(words) - numWords + 1)])
            nWordLens = np.array([len(nWord)+1 for nWord in nWords]).astype(np.float)
            nWordLens[nWordLens > 256] = -np.inf
            if np.all(nWordLens == -np.inf):
                print("Max target length reached. Exiting")
                exit()     

            nWords = nWords[nWordLens > 0]       

            for ix in range(len(nWords)):
                targetNWord = nWords[ix]
                audioStartTime = float(lines[4+ix].split(" ")[1])
                audioEndTime = float(lines[4+ix+numWords-1].split(" ")[2])
                sampFreq, audio = wavfile.read(audioFile)
                inputAudio = audio[int(sampFreq*audioStartTime):int(sampFreq*audioEndTime)]
                inpLen = (len(inputAudio) - 640)//160 + 1
            
                if len(inputAudio) < (640 + 3*160):
                    inpLen = 4
                
                if inpLen > maxInpLen:
                    maxInpLen = inpLen
                    maxTarget = targetNWord

 
    reqLen = required_input_length(maxTarget)
    reqLen = reqLen + 1
    if (reqLen*4 > maxInpLen):
        print(reqLen*4)
    else:
        print(maxInpLen, maxTarget)
    return



def main_max_inplen():

    videoFilesList = list()
    for root, dirs, files in os.walk(args.datadir + "/main"):
        for file in files:
            if file.endswith(".mp4"):
                videoFilesList.append(os.path.join(root, file))

    maxTarget = None
    maxInpLen = 0
    for file in videoFilesList:
        audioFile = file[:-4] + ".wav"
        targetFile = file[:-4] + ".txt"
        with open(targetFile, "r") as f:
            target = f.readline().strip()[7:].replace("'", "")

        sampFreq, audio = wavfile.read(audioFile)
        inpLen = (len(audio) - 640)//160 + 1
        if inpLen > maxInpLen:
            maxInpLen = inpLen
            maxTarget = target

    
    reqLen = required_input_length(maxTarget)
    reqLen = reqLen + 1
    if (reqLen*4 > maxInpLen):
        print(reqLen*4)
    else:
        print(maxInpLen, maxTarget)
    return



if __name__ == '__main__':
    

    if args.check == "input_ge_target":
        input_ge_target()
    
    elif args.check[:20] == "tarlen_distribution_":
        data = args.check[20:]
        target_len_distribution(data=data)

    elif args.check == "word_len_distribution":
        word_len_distribution()

    elif args.check == "max_fcount_chars_ratio":
        max_fcount_chars_ratio()

    elif args.check[:20] == "pretrain_max_inplen_":
        numWords = int(args.check[20:])
        pretrain_max_inplen(numWords=numWords)

    elif args.check == "main_max_inplen":
        main_max_inplen()

    else:
        print("Invalid check")