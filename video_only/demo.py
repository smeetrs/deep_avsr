"""
Specifications:

Videofile - demofile.mp4
Properties - Video:
             25 fps, 160x160 RGB frames, Mouth approx. in center,
             face size should be comparable to frame size
             Audio:
             Mono audio, 16000 Hz sample rate

Targetfile - demofile.txt
Content -
Text:  THIS SENTENCE IS ONLY FOR DEMO PURPOSE A NUMBER LIKE 4 CAN ALSO BE USED
Note - Target length <= 100 characters. All characters in capital and no punctuations other than
       an apostrophe (').

In real world long videos, each video can be appropriately segmented into clips of appropriate length
depending on the speaking rate of the speaker. For a speaker with around 160 words per min,
and 6 characters per word (including space) on average, clip lengths should be around 6 secs.
A prediction concatenating algorithm would be needed to get the final prediction for the complete
video in such cases.
"""

import torch
import numpy as np
import cv2 as cv
import os

from config import args
from models.video_net import VideoNet
from models.visual_frontend import VisualFrontend
from models.lrs2_char_lm import LRS2CharLM
from data.utils import prepare_main_input, collate_fn
from utils.preprocessing import preprocess_sample
from utils.decoders import ctc_greedy_decode, ctc_search_decode
from utils.metrics import compute_cer, compute_wer


np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")


if args["TRAINED_MODEL_FILE"] is not None:

    print("\nTrained Model File: %s" %(args["TRAINED_MODEL_FILE"]))
    print("\nDemo Directory: %s" %(args["DEMO_DIRECTORY"]))


    #declaring the model and loading the trained weights
    model = VideoNet(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
                     args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"])
    model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["TRAINED_MODEL_FILE"], map_location=device))
    model.to(device)


    #declaring the visual frontend module
    vf = VisualFrontend()
    vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"], map_location=device))
    vf.to(device)


    #declaring the language model
    lm = LRS2CharLM()
    lm.load_state_dict(torch.load(args["TRAINED_LM_FILE"], map_location=device))
    lm.to(device)
    if not args["USE_LM"]:
        lm = None


    print("\n\nRunning Demo .... \n")

    #walking through the demo directory and running the model on all video files in it
    for root, dirs, files in os.walk(args["DEMO_DIRECTORY"]):
        for file in files:
            if file.endswith(".mp4"):
                sampleFile = os.path.join(root, file[:-4])
                targetFile = os.path.join(root, file[:-4]) + ".txt"

                #preprocessing the sample
                params = {"roiSize":args["ROI_SIZE"], "normMean":args["NORMALIZATION_MEAN"], "normStd":args["NORMALIZATION_STD"], "vf":vf}
                preprocess_sample(sampleFile, params)

                #converting the data sample into appropriate tensors for input to the model
                visualFeaturesFile = os.path.join(root, file[:-4]) + ".npy"
                videoParams = {"videoFPS":args["VIDEO_FPS"]}
                inp, trgt, inpLen, trgtLen = prepare_main_input(visualFeaturesFile, targetFile, args["MAIN_REQ_INPUT_LENGTH"],
                                                                args["CHAR_TO_INDEX"], videoParams)
                inputBatch, targetBatch, inputLenBatch, targetLenBatch = collate_fn([(inp, trgt, inpLen, trgtLen)])

                #running the model
                inputBatch, targetBatch = (inputBatch.float()).to(device), (targetBatch.int()).to(device)
                inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)
                model.eval()
                with torch.no_grad():
                    outputBatch = model(inputBatch)

                #obtaining the prediction using CTC deocder
                if args["TEST_DEMO_DECODING"] == "greedy":
                    predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch, inputLenBatch, args["CHAR_TO_INDEX"]["<EOS>"])

                elif args["TEST_DEMO_DECODING"] == "search":
                    beamSearchParams = {"beamWidth":args["BEAM_WIDTH"], "alpha":args["LM_WEIGHT_ALPHA"], "beta":args["LENGTH_PENALTY_BETA"],
                                        "threshProb":args["THRESH_PROBABILITY"]}
                    predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch, inputLenBatch, beamSearchParams,
                                                                            args["CHAR_TO_INDEX"][" "], args["CHAR_TO_INDEX"]["<EOS>"], lm)

                else:
                    print("Invalid Decode Scheme")
                    exit()

                #computing CER and WER
                cer = compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
                wer = compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, args["CHAR_TO_INDEX"][" "])

                #converting character indices back to characters
                pred = predictionBatch[:][:-1]
                trgt = targetBatch[:][:-1]
                pred = "".join([args["INDEX_TO_CHAR"][ix] for ix in pred.tolist()])
                trgt = "".join([args["INDEX_TO_CHAR"][ix] for ix in trgt.tolist()])

                #printing the predictions
                print("File: %s" %(file))
                print("Prediction: %s" %(pred))
                print("Target: %s" %(trgt))
                print("CER: %.3f  WER: %.3f" %(cer, wer))
                print("\n")


    print("Demo Completed.\n")


else:
    print("\nPath to trained model file not specified.\n")
