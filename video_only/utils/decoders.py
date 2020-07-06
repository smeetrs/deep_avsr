import torch
import numpy as np
from itertools import groupby


np.seterr(divide="ignore")



def ctc_greedy_decode(outputBatch, inputLenBatch, eosIx, blank=0):

    """
    Greedy search technique for CTC decoding.
    This decoding method selects the most probable character at each time step. This is followed by the usual CTC decoding
    to get the predicted transcription.
    Note: The probability assigned to <EOS> token is added to the probability of the blank token before decoding
    to avoid <EOS> predictions in middle of transcriptions. Once decoded, <EOS> token is appended at last to the
    predictions for uniformity with targets.
    """

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
    """
    Class for a single entry in the beam.
    """
    def __init__(self):
        self.logPrTotal = -np.inf
        self.logPrNonBlank = -np.inf
        self.logPrBlank = -np.inf
        self.logPrText = 0
        self.lmApplied = False
        self.lmState = None
        self.labeling = tuple()



class BeamState:

    """
    Class for the beam.
    """

    def __init__(self, alpha, beta):
        self.entries = dict()
        self.alpha = alpha
        self.beta = beta


    def score(self, entry):
        """
        Function to compute score of each entry in the beam.
        """
        labelingLen = len(entry.labeling)
        if labelingLen == 0:
            score = entry.logPrTotal + self.alpha*entry.logPrText
        else:
            score = (entry.logPrTotal + self.alpha*entry.logPrText)/(labelingLen**self.beta)
        return score


    def sort(self):
        """
        Function to sort all the beam entries in descending order depending on their scores.
        """
        beams = [entry for (key, entry) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=self.score)
        return [x.labeling for x in sortedBeams]



def apply_lm(parentBeam, childBeam, spaceIx, lm):

    """
    Applying the language model to obtain the language model character probabilities at a time step
    given all the previous characters.
    """

    if not (childBeam.lmApplied):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if parentBeam.lmState == None:
            initStateBatch = None
            inputBatch = torch.tensor(spaceIx-1).reshape(1,1)
            inputBatch = inputBatch.to(device)
        else:
            initStateBatch = parentBeam.lmState
            inputBatch = torch.tensor(parentBeam.labeling[-1]-1).reshape(1,1)
            inputBatch = inputBatch.to(device)
        lm.eval()
        with torch.no_grad():
            outputBatch, finalStateBatch = lm(inputBatch, initStateBatch)
        logProbs = outputBatch.squeeze()
        logProb = logProbs[childBeam.labeling[-1]-1]
        childBeam.logPrText = parentBeam.logPrText + logProb
        childBeam.lmApplied = True
        childBeam.lmState = finalStateBatch
    return



def add_beam(beamState, labeling):
    """
    Function to add a new entry to the beam.
    """
    if labeling not in beamState.entries.keys():
        beamState.entries[labeling] = BeamEntry()



def log_add(a, b):
    """
    Addition of log probabilities.
    """
    result = np.log(np.exp(a) + np.exp(b))
    return result



def ctc_search_decode(outputBatch, inputLenBatch, beamSearchParams, spaceIx, eosIx, lm, blank=0):

    """
    Applies the CTC beam search decoding along with a character-level language model.
    Note: The probability assigned to <EOS> token is added to the probability of the blank token before decoding
    to avoid <EOS> predictions in middle of transcriptions. Once decoded, <EOS> token is appended at last to the
    predictions for uniformity with targets.
    """

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

        #initializing the main beam with a single entry having empty prediction
        last = BeamState(alpha, beta)
        labeling = tuple()
        last.entries[labeling] = BeamEntry()
        last.entries[labeling].logPrBlank = 0
        last.entries[labeling].logPrTotal = 0

        #going over all the time steps
        for t in range(maxT):

            #a temporary beam to store all possible predictions (which are extensions of predictions
            #in the main beam after time step t-1) after time step t
            curr = BeamState(alpha, beta)
            #considering only the characters with probability above a certain threshold to speeden up the algo
            prunedChars = np.where(mat[t,:] > np.log(threshProb))[0]

            #keeping only the best predictions in the main beam
            bestLabelings = last.sort()[:beamWidth]

            #going over all the best predictions
            for labeling in bestLabelings:

                #same prediction (either blank or last character repeated)
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


                #extending the best prediction with all characters in the pruned set
                for c in prunedChars:

                    if c == blank:
                        continue

                    #extended prediction
                    newLabeling = labeling + (c,)

                    if (len(labeling) != 0)  and (labeling[-1] == c):
                        logPrNonBlank = mat[t, c] + last.entries[labeling].logPrBlank
                    else:
                        logPrNonBlank = mat[t, c] + last.entries[labeling].logPrTotal

                    add_beam(curr, newLabeling)
                    curr.entries[newLabeling].labeling = newLabeling
                    curr.entries[newLabeling].logPrNonBlank = log_add(curr.entries[newLabeling].logPrNonBlank, logPrNonBlank)
                    curr.entries[newLabeling].logPrTotal = log_add(curr.entries[newLabeling].logPrTotal, logPrNonBlank)

                    #applying language model
                    if lm is not None:
                        apply_lm(curr.entries[labeling], curr.entries[newLabeling], spaceIx, lm)

            #replacing the main beam with the temporary beam having extended predictions
            last = curr

        #output the best prediciton
        bestLabeling = last.sort()[0]
        bestLabeling = list(bestLabeling)
        bestLabeling.append(eosIx)
        preds.extend(bestLabeling)
        predLens.append(len(bestLabeling))

    predictionBatch = torch.tensor(preds).int()
    predictionLenBatch = torch.tensor(predLens).int()
    return predictionBatch, predictionLenBatch
