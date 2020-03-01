import torch
import numpy as np
from itertools import groupby

np.seterr(divide="ignore")



def ctc_greedy_decode(outputBatch, inputLenBatch, eosIx, blank=0):
    outputBatch = outputBatch.cpu()
    inputLenBatch = inputLenBatch.cpu()
    outputBatch[:,:,blank] = torch.log(torch.exp(outputBatch[:,:,blank]) + torch.exp(outputBatch[:,:,eosIx]))
    reqIxs = np.arange(outputBatch.shape[2])
    reqIxs = reqIxs[reqIxs != eosIx]
    outputBatch = outputBatch[:,:,reqIxs]
    
    predCharIxs = torch.argmax(outputBatch, dim=2).T.numpy()
    inpLens = inputLenBatch.numpy()
    preds = list()
    predLens = list()
    for i in range(len(predCharIxs)):
        pred = predCharIxs[i]
        ilen = inpLens[i]
        pred = pred[:ilen]
        pred = np.array([x[0] for x in groupby(pred)])
        pred = pred[pred != blank]
        pred = list(pred)
        pred.append(eosIx)
        preds.extend(pred)
        predLens.append(len(pred))
    predictionBatch = torch.tensor(preds).int()
    predictionLenBatch = torch.tensor(predLens).int()
    return predictionBatch, predictionLenBatch




class BeamEntry:
    def __init__(self):
        self.logPrTotal = -np.inf 
        self.logPrNonBlank = -np.inf 
        self.logPrBlank = -np.inf 
        self.logPrText = 0
        self.lmApplied = False
        self.lmState = None
        self.labeling = tuple()


class BeamState:
    def __init__(self, alpha, beta):
        self.entries = dict()
        self.alpha = alpha
        self.beta = beta

    def score(self, entry):
        labelingLen = len(entry.labeling)
        if labelingLen == 0:
            score = entry.logPrTotal + self.alpha*entry.logPrText
        else:
            score = (entry.logPrTotal + self.alpha*entry.logPrText)/(labelingLen**self.beta)
        return score

    def sort(self):
        beams = [entry for (key, entry) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=self.score)
        return [x.labeling for x in sortedBeams]


def apply_lm(parentBeam, childBeam, spaceIx, lm):
    if not (childBeam.lmApplied):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lm.eval()
        if parentBeam.lmState == None:
            initStateBatch = None
            inputBatch = torch.tensor(spaceIx-1).reshape(1,1)
            inputBatch = inputBatch.to(device)
        else:
            initStateBatch = parentBeam.lmState
            inputBatch = torch.tensor(parentBeam.labeling[-1]-1).reshape(1,1)
            inputBatch = inputBatch.to(device) 
        with torch.no_grad():
            outputBatch, finalStateBatch = lm(inputBatch, initStateBatch)
        logProbs = outputBatch.squeeze()
        logProb = logProbs[childBeam.labeling[-1]-1]
        childBeam.logPrText = parentBeam.logPrText + logProb
        childBeam.lmApplied = True
        childBeam.lmState = finalStateBatch
    return


def add_beam(beamState, labeling):
    if labeling not in beamState.entries.keys():
        beamState.entries[labeling] = BeamEntry()


def log_add(a, b):
    result = np.log(np.exp(a) + np.exp(b))
    return result




def ctc_search_decode(outputBatch, inputLenBatch, beamSearchParams, spaceIx, eosIx, lm=None, blank=0):
    outputBatch = outputBatch.cpu()
    inputLenBatch = inputLenBatch.cpu()
    outputBatch[:,:,blank] = torch.log(torch.exp(outputBatch[:,:,blank]) + torch.exp(outputBatch[:,:,eosIx]))
    reqIxs = np.arange(outputBatch.shape[2])
    reqIxs = reqIxs[reqIxs != eosIx]
    outputBatch = outputBatch[:,:,reqIxs]
    
    beamWidth = beamSearchParams["beamWidth"]
    alpha = beamSearchParams["alpha"]
    beta = beamSearchParams["beta"]
    threshProb = beamSearchParams["threshProb"]

    outLogProbs = outputBatch.transpose(0, 1).numpy()
    inpLens = inputLenBatch.numpy()
    preds = list()
    predLens = list()

    for n in range(len(outLogProbs)):
        mat = outLogProbs[n]
        ilen = inpLens[n]
        mat = mat[:ilen,:]
        maxT, maxC = mat.shape

        last = BeamState(alpha=alpha, beta=beta)
        labeling = tuple()
        last.entries[labeling] = BeamEntry()
        last.entries[labeling].logPrBlank = 0
        last.entries[labeling].logPrTotal = 0

        for t in range(maxT):
            curr = BeamState(alpha=alpha, beta=beta)
            prunedChars = np.where(mat[t,:] > np.log(threshProb))[0]

            bestLabelings = last.sort()[:beamWidth]
            for labeling in bestLabelings:

                if len(labeling) != 0:
                    logPrNonBlank = last.entries[labeling].logPrNonBlank + mat[t, labeling[-1]]
                else:
                    logPrNonBlank = -np.inf

                logPrBlank = last.entries[labeling].logPrTotal + mat[t, blank]

                add_beam(curr, labeling)
                curr.entries[labeling].labeling = labeling
                curr.entries[labeling].logPrNonBlank = log_add(curr.entries[labeling].logPrNonBlank, logPrNonBlank)
                curr.entries[labeling].logPrBlank = log_add(curr.entries[labeling].logPrBlank, logPrBlank)
                curr.entries[labeling].logPrTotal = log_add(curr.entries[labeling].logPrTotal, log_add(logPrBlank, logPrNonBlank))
                curr.entries[labeling].logPrText = last.entries[labeling].logPrText 
                curr.entries[labeling].lmApplied = True 
                curr.entries[labeling].lmState = last.entries[labeling].lmState
                

                for c in prunedChars:

                    if c == blank:
                        continue
                    
                    newLabeling = labeling + (c,)

                    if (len(labeling) != 0)  and (labeling[-1] == c):
                        logPrNonBlank = mat[t, c] + last.entries[labeling].logPrBlank
                    else:
                        logPrNonBlank = mat[t, c] + last.entries[labeling].logPrTotal

                    add_beam(curr, newLabeling)
                    curr.entries[newLabeling].labeling = newLabeling
                    curr.entries[newLabeling].logPrNonBlank = log_add(curr.entries[newLabeling].logPrNonBlank, logPrNonBlank)
                    curr.entries[newLabeling].logPrTotal = log_add(curr.entries[newLabeling].logPrTotal, logPrNonBlank)
                    
                    if lm != None:
                        apply_lm(curr.entries[labeling], curr.entries[newLabeling], spaceIx, lm)

            last = curr

        bestLabeling = last.sort()[0] 
        bestLabeling = list(bestLabeling)
        bestLabeling.append(eosIx)
        preds.extend(bestLabeling)
        predLens.append(len(bestLabeling))

    predictionBatch = torch.tensor(preds).int()
    predictionLenBatch = torch.tensor(predLens).int()
    return predictionBatch, predictionLenBatch


