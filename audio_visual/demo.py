import torch
import torch.nn as nn
import numpy as np
import os

from config import args
from models.av_net import AVNet
from models.lrs2_char_lm import LRS2CharLM
from data.utils import prepare_main_input, collate_fn
from utils.decoders import ctc_greedy_decode, ctc_search_decode


np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")



print("\nRunning Demo .... \n")
print("Trained Model File: %s\n" %(args["TRAINED_MODEL_FILE"]))
print("Demo Directory: %s\n\n" %(args["CODE_DIRECTORY"] + "/demo"))


model = AVNet(dModel=args["TX_NUM_FEATURES"], nHeads=args["TX_ATTENTION_HEADS"], 
              numLayers=args["TX_NUM_LAYERS"], peMaxLen=args["PE_MAX_LENGTH"], 
              inSize=args["AUDIO_FEATURE_SIZE"], fcHiddenSize=args["TX_FEEDFORWARD_DIM"], 
              dropout=args["TX_DROPOUT"], numClasses=args["NUM_CLASSES"])
model.to(device)
model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["TRAINED_MODEL_FILE"]))
model.to(device)
model.eval()


for root, dirs, files in os.walk(args["CODE_DIRECTORY"] + "/demo"):
    for file in files:
        if file.endswith(".mp4"):
            videoFile = os.path.join(root, file)
            audioFile = os.path.join(root, file[:-4]) + ".wav"
            roiFile = os.path.join(root, file[:-4]) + ".png"
            targetFile = os.path.join(root, file[:-4]) + ".txt"

            v2aCommand = "ffmpeg -y -v quiet -i " + videoFile + " -ac 1 -ar 16000 -vn " + audioFile
            os.system(v2aCommand)
            
            roiSize = args["ROI_SIZE"]
            captureObj = cv.VideoCapture(videoFile)
            roiSequence = np.empty((roiSize,0), dtype=np.int)
            while (captureObj.isOpened()):
                ret, frame = captureObj.read()
                if ret == True:
                    frame = cv.resize(frame, (224,224), interpolation=cv.INTER_CUBIC)
                    grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    roi = grayed[int(112-(roiSize/2)):int(112+(roiSize/2)), int(112-(roiSize/2)):int(112+(roiSize/2))]
                    roiSequence = np.hstack((roiSequence, roi))
                else:
                    break
            captureObj.release()
            cv.imwrite(roiFile, roiSequence)


            stftParams = {"window":args["STFT_WINDOW"], "winLen":args["STFT_WIN_LENGTH"], "overlap":args["STFT_OVERLAP"]}
            videoParams = {"videoFPS":args["VIDEO_FPS"], "roiSize":args["ROI_SIZE"], "normMean":args["NORMALIZATION_MEAN"], 
                           "normStd":args["NORMALIZATION_STD"]}
            inp, trgt, inpLen, trgtLen = prepare_main_input(audioFile, targetFile, args["CHAR_TO_INDEX"], 
                                                            stftParams, videoParams)
            inputBatch, targetBatch, inputLenBatch, targetLenBatch = collate_fn([(inp, trgt, inpLen, trgtLen)])

            inputBatch, targetBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device)), 
                                      (targetBatch.int()).to(device)
            inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)
            with torch.no_grad():
                outputBatch = model(inputBatch)
            
            if args["TEST_DEMO_DECODING"] == "greedy":
                predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch, inputLenBatch, 
                                                                        eosIx=args["CHAR_TO_INDEX"]["<EOS>"])
            elif args["TEST_DEMO_DECODING"] == "search":
                if args["USE_LM"]:
                    lm = LRS2CharLM().to(device)
                    lm.load_state_dict(torch.load(args["TRAINED_LM_FILE"]))
                    lm.to(device)
                else:
                    lm = None

                beamSearchParams={"beamWidth":args["BEAM_WIDTH"], "alpha":args["LM_WEIGHT_ALPHA"], "beta":args["LENGTH_PENALTY_BETA"], 
                                  "threshProb":args["THRESH_PROBABILITY"]}  
                predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch, inputLenBatch,
                                                                        beamSearchParams=beamSearchParams,  
                                                                        spaceIx=args["CHAR_TO_INDEX"][" "],
                                                                        eosIx=args["CHAR_TO_INDEX"]["<EOS>"], 
                                                                        lm=lm)
            else:
                print("Invalid Decode Scheme")
                exit()

            pred = predictionBatch[:][:-1]
            trgt = targetBatch[:][:-1]
            pred = "".join([args["INDEX_TO_CHAR"][ix] for ix in pred.tolist()])
            trgt = "".join([args["INDEX_TO_CHAR"][ix] for ix in trgt.tolist()])
        
            print("File: %s" %(file))
            print("Prediction: %s" %(pred))
            print("Target: %s" %(trgt))
            print("\n")


print("Demo Completed.\n")
