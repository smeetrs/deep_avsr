import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, shutil

from config import args
from models.video_net import VideoNet
from data.lrs2_dataset import LRS2Main
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


#declaring the train and validation datasets and their corresponding dataloaders
videoParams={"videoFPS":args["VIDEO_FPS"]}
trainData = LRS2Main(dataset="train", datadir=args["DATA_DIRECTORY"], reqInpLen=args["MAIN_REQ_INPUT_LENGTH"], 
                     charToIx=args["CHAR_TO_INDEX"], stepSize=args["STEP_SIZE"], videoParams=videoParams)
valData = LRS2Main(dataset="val", datadir=args["DATA_DIRECTORY"], reqInpLen=args["MAIN_REQ_INPUT_LENGTH"], 
                   charToIx=args["CHAR_TO_INDEX"], stepSize=args["STEP_SIZE"], videoParams=videoParams)
trainLoader = DataLoader(trainData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)
valLoader = DataLoader(valData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)


#declaring the model, optimizer, scheduler and the loss function
model = VideoNet(dModel=args["TX_NUM_FEATURES"], nHeads=args["TX_ATTENTION_HEADS"], numLayers=args["TX_NUM_LAYERS"], 
                 peMaxLen=args["PE_MAX_LENGTH"], fcHiddenSize=args["TX_FEEDFORWARD_DIM"], dropout=args["TX_DROPOUT"], 
                 numClasses=args["NUM_CLASSES"])
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args["INIT_LR"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args["LR_SCHEDULER_FACTOR"], 
                                                 patience=args["LR_SCHEDULER_WAIT"], threshold=args["LR_SCHEDULER_THRESH"], 
                                                 threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)
loss_function = nn.CTCLoss(blank=0, zero_infinity=False)


#removing the checkpoints directory if it exists and remaking it
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


#loading the pretrained weights
if args["PRETRAINED_MODEL_FILE"] is not None:

    print("\n\nPre-trained Model File: %s" %(args["PRETRAINED_MODEL_FILE"]))
    print("\nLoading the pre-trained model .... \n")
    model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["PRETRAINED_MODEL_FILE"]))
    model.to(device)
    print("Loading Done.\n")    



trainingLossCurve = list()
validationLossCurve = list()
trainingWERCurve = list()
validationWERCurve = list()


print("\nTraining the model .... \n")

#printing the total and trainable parameters in the model
numTotalParams, numTrainableParams = num_params(model)
print("Number of total parameters in the model = %d" %(numTotalParams))
print("Number of trainable parameters in the model = %d\n" %(numTrainableParams))

trainParams = {"spaceIx":args["CHAR_TO_INDEX"][" "], "eosIx":args["CHAR_TO_INDEX"]["<EOS>"]}
valParams = {"decodeScheme":"greedy", "spaceIx":args["CHAR_TO_INDEX"][" "], "eosIx":args["CHAR_TO_INDEX"]["<EOS>"]}


for step in range(1, args["NUM_STEPS"]+1):
    
    #train the model for one step
    trainingLoss, trainingCER, trainingWER = train(model, trainLoader, optimizer, loss_function, device, trainParams)
    trainingLossCurve.append(trainingLoss)
    trainingWERCurve.append(trainingWER)

    #evaluate the model on validation set
    validationLoss, validationCER, validationWER = evaluate(model, valLoader, loss_function, device, valParams)
    validationLossCurve.append(validationLoss)
    validationWERCurve.append(validationWER)

    #make a scheduler step
    scheduler.step(validationCER)

    #printing the stats after each step
    print("Step: %d || Tr.Loss: %.6f || Val.Loss: %.6f || Tr.CER: %.3f || Val.CER: %.3f || Tr.WER: %.3f || Val.WER: %.3f" 
          %(step, trainingLoss, validationLoss, trainingCER, validationCER, trainingWER, validationWER))
    

    #saving the model weights and loss/metric curves in the checkpoints directory after every few steps
    if (step % args["SAVE_FREQUENCY"] == 0) or (step == args["NUM_STEPS"]):
        
        savePath = args["CODE_DIRECTORY"] + "/checkpoints/models/train-step_{:04d}-wer_{:.3f}.pt".format(step, validationWER)
        torch.save(model.state_dict(), savePath)

        plt.figure()
        plt.title("Loss Curves")
        plt.xlabel("Step No.")
        plt.ylabel("Loss value")
        plt.plot(list(range(1, len(trainingLossCurve)+1)), trainingLossCurve, "blue", label="Train")
        plt.plot(list(range(1, len(validationLossCurve)+1)), validationLossCurve, "red", label="Validation")
        plt.legend()
        plt.savefig(args["CODE_DIRECTORY"] + "/checkpoints/plots/train-step_{:04d}-loss.png".format(step))
        plt.close()

        plt.figure()
        plt.title("WER Curves")
        plt.xlabel("Step No.")
        plt.ylabel("WER")
        plt.plot(list(range(1, len(trainingWERCurve)+1)), trainingWERCurve, "blue", label="Train")
        plt.plot(list(range(1, len(validationWERCurve)+1)), validationWERCurve, "red", label="Validation")
        plt.legend()
        plt.savefig(args["CODE_DIRECTORY"] + "/checkpoints/plots/train-step_{:04d}-wer.png".format(step))
        plt.close()

    #empty cache after every step to avoid filling up the GPU memory when using variable sized batches
    if args["EMPTY_CACHE"]:
        if torch.cuda.is_available():
           torch.cuda.empty_cache() 


print("\nTraining Done.\n")
