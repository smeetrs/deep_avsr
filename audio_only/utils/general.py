import torch
import numpy as np

from .metrics import compute_cer, compute_wer
from .decoders import ctc_greedy_decode, ctc_search_decode


def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams



def train(model, trainLoader, optimizer, loss_function, device, trainParams):

    """
    Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
    It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
    """

    model.train()
    trainingLoss = 0
    trainingCER = 0
    trainingWER = 0

    for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(trainLoader):
        
        inputBatch, targetBatch = (inputBatch.float()).to(device), (targetBatch.int()).to(device)
        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)
        
        optimizer.zero_grad()
        outputBatch = model(inputBatch)
        loss = loss_function(outputBatch, targetBatch, inputLenBatch, targetLenBatch)
        loss.backward()
        optimizer.step()

        trainingLoss = trainingLoss + loss.item()
        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch.detach(), inputLenBatch, 
                                                                eosIx=trainParams["eosIx"])
        trainingCER = trainingCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        trainingWER = trainingWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, 
                                                spaceIx=trainParams["spaceIx"])
    
    trainingLoss = trainingLoss/len(trainLoader)
    trainingCER = trainingCER/len(trainLoader)
    trainingWER = trainingWER/len(trainLoader)
    return trainingLoss, trainingCER, trainingWER



def evaluate(model, evalLoader, loss_function, device, evalParams):

    """
    Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
    The CTC decode scheme can be set to either 'greedy' or 'search'. 
    """

    model.eval()
    evalLoss = 0
    evalCER = 0
    evalWER = 0
    
    with torch.no_grad():
        for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(evalLoader):
            
            inputBatch, targetBatch = (inputBatch.float()).to(device), (targetBatch.int()).to(device)
            inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)
            
            outputBatch = model(inputBatch)
            loss = loss_function(outputBatch, targetBatch, inputLenBatch, targetLenBatch)

            evalLoss = evalLoss + loss.item()
            if evalParams["decodeScheme"] == "greedy":
                predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch, inputLenBatch,
                                                                        eosIx=evalParams["eosIx"])
            elif evalParams["decodeScheme"] == "search":
                predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch, inputLenBatch,
                                                                        beamSearchParams=evalParams["beamSearchParams"],  
                                                                        spaceIx=evalParams["spaceIx"], 
                                                                        eosIx=evalParams["eosIx"], lm=evalParams["lm"])
            else:
                print("Invalid Decode Scheme")
                exit()
            
            evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
            evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch,
                                            spaceIx=evalParams["spaceIx"])

    evalLoss = evalLoss/len(evalLoader)
    evalCER = evalCER/len(evalLoader)
    evalWER = evalWER/len(evalLoader)
    return evalLoss, evalCER, evalWER

