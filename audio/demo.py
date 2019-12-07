import torch
import torch.nn as nn
import numpy as np
import os

from config import args
from models.audio_net import AudioNet
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


model = AudioNet(dModel=args["TX_NUM_FEATURES"], nHeads=args["TX_ATTENTION_HEADS"], 
                 numLayers=args["TX_NUM_LAYERS"], peMaxLen=args["PE_MAX_LENGTH"], 
                 inSize=args["AUDIO_FEATURE_SIZE"], fcHiddenSize=args["TX_FEEDFORWARD_DIM"], 
                 dropout=args["TX_DROPOUT"], numClasses=args["NUM_CLASSES"])
model.to(device)
model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["TRAINED_MODEL_FILE"]))
model.to(device)
model.eval()


for root, dirs, files in os.walk(args["CODE_DIRECTORY"] + "/demo"):
    for file in files:
        if file.endswith(".wav"):
            audioFile = os.path.join(root, file)
            targetFile = os.path.join(root, file[:-4]) + ".txt"

            stftParams={"window":args["STFT_WINDOW"], "winLen":args["STFT_WIN_LENGTH"], "overlap":args["STFT_OVERLAP"]}
            inp, trgt, inpLen, trgtLen = prepare_main_input(audioFile, targetFile, args["CHAR_TO_INDEX"], stftParams)
            inputBatch, targetBatch, inputLenBatch, targetLenBatch = collate_fn([(inp, trgt, inpLen, trgtLen)])

            inputBatch, targetBatch = (inputBatch.float()).to(device), (targetBatch.int()).to(device)
            inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)
            with torch.no_grad():
                outputBatch = model(inputBatch)
            
            if args["TEST_DEMO_DECODING"] == "greedy":
                predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch, inputLenBatch)
            elif args["TEST_DEMO_DECODING"] == "search":
                lm = LRS2CharLM().to(device)
                lm.load_state_dict(torch.load(args["TRAINED_LM_FILE"]))
                lm.to(device)
                predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch, inputLenBatch,
                                                                        beamSearchParams={"beamWidth":args["BEAM_WIDTH"], 
                                                                                          "alpha":args["LM_WEIGHT_ALPHA"], 
                                                                                          "beta":args["LENGTH_PENALTY_BETA"]},  
                                                                        spaceIx=args["CHAR_TO_INDEX"][" "], 
                                                                        lm=lm)
            else:
                print("Invalid Decode Scheme")
                exit()

            pred = predictionBatch[:]
            pred = "".join([args["INDEX_TO_CHAR"][ix] for ix in pred.tolist()])
            trgt = "".join([args["INDEX_TO_CHAR"][ix] for ix in trgt.tolist()])
        
            print("File: %s" %(file))
            print("Prediction: %s" %(pred))
            print("Target: %s" %(trgt))
            print("\n")


print("Demo Completed.\n")
