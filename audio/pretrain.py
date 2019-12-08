import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, shutil

from config import args
from models.audio_net import AudioNet
from models.lrs2_char_lm import LRS2CharLM
from data.lrs2_dataset import LRS2Pretrain
from data.utils import collate_fn
from utils.general import num_params, train, evaluate



matplotlib.use("Agg")
np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")
kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


pretrainData = LRS2Pretrain(datadir=args["DATA_DIRECTORY"], numWords=args["PRETRAIN_NUM_WORDS"], 
                            charToIx=args["CHAR_TO_INDEX"], stepSize=args["STEP_SIZE"], 
                            stftParams={"window":args["STFT_WINDOW"], 
                                        "winLen":args["STFT_WIN_LENGTH"], 
                                        "overlap":args["STFT_OVERLAP"]})
pretrainValSize = int(args["PRETRAIN_VAL_SPLIT"]*len(pretrainData))
pretrainSize = len(pretrainData) - pretrainValSize
pretrainData, pretrainValData = random_split(pretrainData, [pretrainSize, pretrainValSize])
pretrainLoader = DataLoader(pretrainData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)
pretrainValLoader = DataLoader(pretrainValData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)



model = AudioNet(dModel=args["TX_NUM_FEATURES"], nHeads=args["TX_ATTENTION_HEADS"], 
                 numLayers=args["TX_NUM_LAYERS"], peMaxLen=args["PE_MAX_LENGTH"], 
                 inSize=args["AUDIO_FEATURE_SIZE"], fcHiddenSize=args["TX_FEEDFORWARD_DIM"], 
                 dropout=args["TX_DROPOUT"], numClasses=args["NUM_CLASSES"])
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args["INIT_LR"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args["LR_SCHEDULER_FACTOR"], 
                                                 patience=args["LR_SCHEDULER_WAIT"], threshold=args["LR_SCHEDULER_THRESH"], 
                                                 threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)
loss_function = nn.CTCLoss(blank=0, zero_infinity=False)



if os.path.exists(args["CODE_DIRECTORY"] + "/checkpoints"):
    
    while True:
        ch = input("Continue and remove the 'checkpoints' directory? y/n: ")
        if ch == "y":
            break
        elif ch == "n":
            exit()
        else:
            print("Invalid input")
    shutil.rmtree(args["CODE_DIRECTORY"] + "/checkpoints")

os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints")
os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints/models")
os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints/plots")



if args["PRETRAINED_MODEL_FILE"] is not None:

    print("\n\nPre-trained Model File: %s" %(args["PRETRAINED_MODEL_FILE"]))
    print("\nLoading the pre-trained model .... \n")
    model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["PRETRAINED_MODEL_FILE"]))
    model.to(device)
    print("\nLoading Done.\n")    



trainingLossCurve = list()
validationLossCurve = list()
trainingWERCurve = list()
validationWERCurve = list()


print("\nPretraining the model .... \n")

numTotalParams, numTrainableParams = num_params(model)
print("Number of total parameters in the model = %d" %(numTotalParams))
print("Number of trainable parameters in the model = %d\n" %(numTrainableParams))

trainParams = {"spaceIx":args["CHAR_TO_INDEX"][" "]}
lm = LRS2CharLM().to(device)
lm.load_state_dict(torch.load(args["TRAINED_LM_FILE"]))
lm.to(device)
valParams = {"decodeScheme":"greedy",
              "beamSearchParams":{"beamWidth":args["BEAM_WIDTH"], 
                                  "alpha":args["LM_WEIGHT_ALPHA"], 
                                  "beta":args["LENGTH_PENALTY_BETA"],
                                  "threshProb":args["THRESH_PROBABILITY"]},
              "spaceIx":args["CHAR_TO_INDEX"][" "],
              "lm":lm}


for step in range(1, args["NUM_STEPS"]+1):
    
    trainingLoss, trainingCER, trainingWER = train(model, pretrainLoader, optimizer, loss_function, device, trainParams)
    trainingLossCurve.append(trainingLoss)
    trainingWERCurve.append(trainingWER)

    validationLoss, validationCER, validationWER = evaluate(model, pretrainValLoader, loss_function, device, valParams)
    validationLossCurve.append(validationLoss)
    validationWERCurve.append(validationWER)

    scheduler.step(validationWER)

    print("Step: %d || Tr.Loss: %.6f || Val.Loss: %.6f || Tr.CER: %.3f || Val.CER: %.3f || Tr.WER: %.3f || Val.WER: %.3f" 
          %(step, trainingLoss, validationLoss, trainingCER, validationCER, trainingWER, validationWER))
    

    if (step % args["SAVE_FREQUENCY"] == 0) or (step == args["NUM_STEPS"]):
        
        savePath = args["CODE_DIRECTORY"] + "/checkpoints/models/pretrain_{:03d}w-step_{:04d}-wer_{:.3f}.pt".format(args["PRETRAIN_NUM_WORDS"],
                                                                                                                    step, validationWER)
        torch.save(model.state_dict(), savePath)

        plt.figure()
        plt.title("Loss Curves")
        plt.xlabel("Step No.")
        plt.ylabel("Loss value")
        plt.plot(list(range(1, len(trainingLossCurve)+1)), trainingLossCurve, "blue", label="Train")
        plt.plot(list(range(1, len(validationLossCurve)+1)), validationLossCurve, "red", label="Validation")
        plt.legend()
        plt.savefig(args["CODE_DIRECTORY"] + "/checkpoints/plots/pretrain_{:03d}w-step_{:04d}-loss.png".format(args["PRETRAIN_NUM_WORDS"], step))
        plt.close()

        plt.figure()
        plt.title("WER Curves")
        plt.xlabel("Step No.")
        plt.ylabel("WER")
        plt.plot(list(range(1, len(trainingWERCurve)+1)), trainingWERCurve, "blue", label="Train")
        plt.plot(list(range(1, len(validationWERCurve)+1)), validationWERCurve, "red", label="Validation")
        plt.legend()
        plt.savefig(args["CODE_DIRECTORY"] + "/checkpoints/plots/pretrain_{:03d}w-step_{:04d}-wer.png".format(args["PRETRAIN_NUM_WORDS"], step))
        plt.close()


    if args["EMPTY_CACHE"]:
        if torch.cuda.is_available():
           torch.cuda.empty_cache() 


print("\nPretraining Done.\n")
