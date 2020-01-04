import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from config import args
from models.audio_net import AudioNet
from models.lrs2_char_lm import LRS2CharLM
from data.lrs2_dataset import LRS2Main
from data.utils import collate_fn
from utils.general import evaluate



np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")
kwargs = {"num_workers":args["NUM_WORKERS"], "pin_memory":True} if gpuAvailable else {}
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


testData = LRS2Main(dataset="test", datadir=args["DATA_DIRECTORY"], charToIx=args["CHAR_TO_INDEX"], 
                    stepSize=args["STEP_SIZE"], stftParams={"window":args["STFT_WINDOW"], 
                                                            "winLen":args["STFT_WIN_LENGTH"], 
                                                            "overlap":args["STFT_OVERLAP"]})
testLoader = DataLoader(testData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)


model = AudioNet(dModel=args["TX_NUM_FEATURES"], nHeads=args["TX_ATTENTION_HEADS"], 
                 numLayers=args["TX_NUM_LAYERS"], peMaxLen=args["PE_MAX_LENGTH"], 
                 inSize=args["AUDIO_FEATURE_SIZE"], fcHiddenSize=args["TX_FEEDFORWARD_DIM"], 
                 dropout=args["TX_DROPOUT"], numClasses=args["NUM_CLASSES"])
model.to(device)
loss_function = nn.CTCLoss(blank=0, zero_infinity=False)



if args["TRAINED_MODEL_FILE"] is not None:

    print("\nTrained Model File: %s" %(args["TRAINED_MODEL_FILE"]))
    print("\nTesting the trained model .... \n")

    model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["TRAINED_MODEL_FILE"]))
    model.to(device)

    lm = LRS2CharLM().to(device)
    lm.load_state_dict(torch.load(args["TRAINED_LM_FILE"]))
    lm.to(device)
    beamSearchParams = {"beamWidth":args["BEAM_WIDTH"], "alpha":args["LM_WEIGHT_ALPHA"], "beta":args["LENGTH_PENALTY_BETA"], "threshProb":args["THRESH_PROBABILITY"]}
    testParams = {"decodeScheme":args["TEST_DEMO_DECODING"], "beamSearchParams":beamSearchParams, "spaceIx":args["CHAR_TO_INDEX"][" "], "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "lm":lm}
    testLoss, testCER, testWER = evaluate(model, testLoader, loss_function, device, testParams)
    
    print("Test Loss: %.6f || Test CER: %.3f || Test WER: %.3f" %(testLoss, testCER, testWER))
    print("\nTesting Done.\n")    
