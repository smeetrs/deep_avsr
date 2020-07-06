import torch.nn as nn
import torch.nn.functional as F



class LRS2CharLM(nn.Module):

    """
    A character-level language model for the LRS2 Dataset.
    Architecture: Unidirectional 4-layered 1024-dim LSTM model
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe (''), space ( )
    Output: Log probabilities over the character set
    Note: The space character plays the role of the start-of-sequence token as well.
    """

    def __init__(self):
        super(LRS2CharLM, self).__init__()
        self.embedding = nn.Embedding(38, 1024, padding_idx=None)
        self.lstm = nn.LSTM(1024, 1024, num_layers=4)
        self.fc = nn.Linear(1024, 38)
        return


    def forward(self, inputBatch, initStateBatch):
        batch = self.embedding(inputBatch)
        if initStateBatch != None:
            batch, finalStateBatch = self.lstm(batch, initStateBatch)
        else:
            batch, finalStateBatch = self.lstm(batch)
        batch = batch.transpose(0, 1)
        outputBatch = F.log_softmax(self.fc(batch), dim=2)
        outputBatch = outputBatch.transpose(0, 1)
        return outputBatch, finalStateBatch
