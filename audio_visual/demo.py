import torch
import torch.nn as nn
import numpy as np
import os

from config import args
from models.av_net import AVNet
from models.lrs2_char_lm import LRS2CharLM
from models.visual_frontend import VisualFrontend
from data.utils import prepare_main_input, collate_fn
from utils.preprocessing import preprocess_sample
from utils.decoders import ctc_greedy_decode, ctc_search_decode


np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")



print("\nRunning Demo .... \n")
print("Trained Model File: %s\n" %(args["TRAINED_MODEL_FILE"]))
print("Demo Directory: %s\n\n" %(args["CODE_DIRECTORY"] + "/demo"))


#declaring the model and loading the trained weights
model = AVNet(dModel=args["TX_NUM_FEATURES"], nHeads=args["TX_ATTENTION_HEADS"], 
              numLayers=args["TX_NUM_LAYERS"], peMaxLen=args["PE_MAX_LENGTH"], 
              inSize=args["AUDIO_FEATURE_SIZE"], fcHiddenSize=args["TX_FEEDFORWARD_DIM"], 
              dropout=args["TX_DROPOUT"], numClasses=args["NUM_CLASSES"])
model.to(device)
model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["TRAINED_MODEL_FILE"]))
model.to(device)
model.eval()

#declaring the visual frontend module
vf = VisualFrontend().to(device)
vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"]))
vf.to(device)
vf.eval()


#walking through the demo directory and running the model on all video files in it 
for root, dirs, files in os.walk(args["CODE_DIRECTORY"] + "/demo"):
    for file in files:
        if file.endswith(".mp4"):
            sampleFile = os.path.join(root, file[:-4])
            targetFile = os.path.join(root, file[:-4]) + ".txt"

            #preprocessing the sample
            params = {"roiSize":args["ROI_SIZE"], "normMean":args["NORMALIZATION_MEAN"], "normStd":args["NORMALIZATION_STD"], "vf":vf}
            preprocess_sample(sampleFile, params)

            #converting the data sample into appropriate tensors for input to the model
            audioFile = os.path.join(root, file[:-4]) + ".wav"
            visualFeaturesFile = os.path.join(root, file[:-4]) + ".npy"
            audioParams = {"stftWindow":args["STFT_WINDOW"], "stftWinLen":args["STFT_WIN_LENGTH"], "stftOverlap":args["STFT_OVERLAP"]}
            videoParams = {"videoFPS":args["VIDEO_FPS"]}
            inp, trgt, inpLen, trgtLen = prepare_main_input(audioFile, visualFeaturesFile, targetFile, None, args["CHAR_TO_INDEX"], 
                                                            args["NOISE_SNR_DB"], audioParams, videoParams)
            inputBatch, targetBatch, inputLenBatch, targetLenBatch = collate_fn([(inp, trgt, inpLen, trgtLen)])

            #running the model
            inputBatch, targetBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device)), (targetBatch.int()).to(device)
            inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)
            with torch.no_grad():
                outputBatch = model(inputBatch)
            
            #obtaining the prediction using CTC deocder
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

            #converting character indices back to characters
            pred = predictionBatch[:][:-1]
            trgt = targetBatch[:][:-1]
            pred = "".join([args["INDEX_TO_CHAR"][ix] for ix in pred.tolist()])
            trgt = "".join([args["INDEX_TO_CHAR"][ix] for ix in trgt.tolist()])
            
            #printing the predictions
            print("File: %s" %(file))
            print("Prediction: %s" %(pred))
            print("Target: %s" %(trgt))
            print("\n")


print("Demo Completed.\n")
