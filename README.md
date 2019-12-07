# Deep Audio-Visual Speech Recognition

The repository contains a PyTorch reproduction of the [Deep Audio-Visual Speech Recognition](https://arxiv.org/abs/1809.02108) paper.


## Changes Required
- saving best models by observing the val wer instead of at regular intervals
- removing the non-word regions in the data by observing the time
- retraining/fine-tuning the models with apostrophe and ordering according to lang model (with no eos and pad)(shuffle the output layer and fine-tune) 
- going through github code
- using ctc decoders written in c++
- commenting the code (do it last since editing the comments is a PITA)
- training the 33 word and 38 word
- testing all parts of the new code