# Deep Audio-Visual Speech Recognition

The repository contains a PyTorch reproduction of the [Deep Audio-Visual Speech Recognition](https://arxiv.org/abs/1809.02108) paper.


## Changes Required
- Adding ignored files to github, unsetting git credentials
- commenting the code (do it last since editing the comments is a PITA)
- av training - visual init, 0.5 noise prob, no noise in val and test sets, keep encoders fixed, training: 3 rounds - 0.5 noise prob, encoders fixed - 1 noise prob, audio encoder fixed, finetune - 0 noise prob, nothing fixed, finetune, target wer greedy = 14%, search = 9%
- deleting all wav and png files from dataset at last
- noisy training (audio/video/both) and data augmentation at the end if needed