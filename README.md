# Deep Audio-Visual Speech Recognition

The repository contains a PyTorch reproduction of the [Deep Audio-Visual Speech Recognition](https://arxiv.org/abs/1809.02108) paper.


## Changes Required
- Adding ignored files to github, unsetting git credentials
- removing the non-word regions in the data by observing the time
- going through github code
- commenting the code (do it last since editing the comments is a PITA)
- video - pretrain 1w three models, one with only pretrained resnet, one with pretrained resnet and encoder and fixing them, previous case but not fixed
- noisy audio and av training
- deleting all wav and png files from dataset at last