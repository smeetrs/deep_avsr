import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import os

from config import args
from models.video_net import VideoNet
from models.visual_frontend import VisualFrontend
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


model = VideoNet(dModel=args["TX_NUM_FEATURES"], nHeads=args["TX_ATTENTION_HEADS"], numLayers=args["TX_NUM_LAYERS"], 
                 peMaxLen=args["PE_MAX_LENGTH"], fcHiddenSize=args["TX_FEEDFORWARD_DIM"], dropout=args["TX_DROPOUT"], 
                 numClasses=args["NUM_CLASSES"])
model.to(device)
model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["TRAINED_MODEL_FILE"]))
model.to(device)
model.eval()


vf = VisualFrontend().to(device)
vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"]))
vf.to(device)
vf.eval()


for root, dirs, files in os.walk(args["CODE_DIRECTORY"] + "/demo"):
    for file in files:
        if file.endswith(".mp4"):
            videoFile = os.path.join(root, file)
            roiFile = os.path.join(root, file[:-4]) + ".png"
            visualFeaturesFile = os.path.join(root, file[:-4]) + ".npy"
            targetFile = os.path.join(root, file[:-4]) + ".txt"
            
            roiSize = args["ROI_SIZE"]
            normMean = args["NORMALIZATION_MEAN"]
            normStd = args["NORMALIZATION_STD"]

            captureObj = cv.VideoCapture(videoFile)
            roiSequence = list()
            while (captureObj.isOpened()):
                ret, frame = captureObj.read()
                if ret == True:
                    grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    grayed = grayed/255
                    grayed = cv.resize(grayed, (224,224))
                    roi = grayed[int(112-(roiSize/2)):int(112+(roiSize/2)), int(112-(roiSize/2)):int(112+(roiSize/2))]
                    roiSequence.append(roi)
                else:
                    break
            captureObj.release()
            cv.imwrite(roiFile, np.floor(255*np.hstack(roiSequence)).astype(np.int))
            
            roiSequence = [roi.reshape((roi.shape[0],roi.shape[1],1)) for roi in roiSequence]
            inp = np.dstack(roiSequence)
            inp = np.transpose(inp, (2,0,1))
            inp = inp.reshape((inp.shape[0],1,1,inp.shape[1],inp.shape[2]))
            inp = (inp - normMean)/normStd
            inputBatch = torch.from_numpy(inp)
            inputBatch = (inputBatch.float()).to(device)
            with torch.no_grad():
                outputBatch = vf(inputBatch)
            out = outputBatch.view(outputBatch.size(0), outputBatch.size(2))
            out = out.cpu().numpy()
            np.save(visualFeaturesFile, out)
            

            videoParams = {"videoFPS":args["VIDEO_FPS"]}
            inp, trgt, inpLen, trgtLen = prepare_main_input(visualFeaturesFile, targetFile, args["CHAR_TO_INDEX"], videoParams)
            inputBatch, targetBatch, inputLenBatch, targetLenBatch = collate_fn([(inp, trgt, inpLen, trgtLen)])

            inputBatch, targetBatch = (inputBatch.float()).to(device), (targetBatch.int()).to(device)
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
