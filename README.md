# Deep Audio-Visual Speech Recognition

The repository contains a PyTorch reproduction of the [Deep Audio-Visual Speech Recognition](https://arxiv.org/abs/1809.02108) paper.

## Requirements

System packages:

	ffmpeg

Python packages:

	editdistance==0.5.3
	matplotlib==3.1.1
	numpy==1.18.1
	opencv-python==4.2.0.32
	pytorch==1.2.0
	scipy==1.3.1
	tqdm==4.42.1

CUDA 10.0 (optional):

	cudatoolkit==10.0

## Project Structure

#### Directories

`/checkpoints`: Directory to store intermediate model weights and plots while training.

`/data`: Directory containing dataset-related files. Specifically, it contains the LRS2 main and pretrain dataset class definitions and other required utility functions.

`/demo`: Directory containing data samples from LRS2 dataset for demo.

`/final`: Directory to store the final trained model weights and plots.

`/models`: Directory containing the transformer model and the language model class definitions.

`/utils`: Directory for all utility functions. Specifically, it contains function definitions for calculating CER and WER, greedy search and beam search CTC decoders and preprocessing of data samples. It also contains general training and evaluating functions.

#### Files

`checker.py`: File containing checker/debug functions for testing all the modules and the functions in the project as well as any other checks to be performed.

`config.py`: File to set the configuration options and the hyperparameter values.

`demo.py`: Python script for running a demo of the trained model on all the data samples in the `demo` directory.

`preprocess.py`: Python script for preprocessing all the data samples in the dataset.

 `pretrain.py`: Python script for pretraining the model on the pretrain dataset using curriculum learning.

`test.py`: Python script to test the trained model on the test set.

`train.py`: Python script to train the model on the training dataset.

## How To Use

#### Training

1. Setting the configuration options in the `config.py` file.
2. Running the `preprocess.py` file.
3. Running the `pretrain.py` file.
4. Running the `train.py` file.
5. Running the `test.py` file.
6. Running the `demo.py` file.

#### Inference

1. Setting the configuration options in the `config.py` file.
2. Running the `demo.py` file.

## Pre-trained Weights

Will be made available soon.

## Results

The Word Error Rates achieved by the models on the test set of the LRS2-BBC dataset are as follows:

| Model        | CTC Greedy Search | CTC Beam Search + LM |
| ------------ | ----------------- | -------------------- |
| Audio-only   | 15.5%             | 11.7%                |
| Video-only   | 72.1%             | 62.7%                |
| Audio-visual | 14.3%             | 10.5%                 |

The WER of the Audio-visual model on the test set of LRS2-BBC dataset using CTC Beam Search + Language Model given in the paper is 8.2%. Since the authors of the paper pretrain the model using the LRS3-TED and MV-LRS datasets as well, the WER we achieved would be higher than 8.2%. However, with some minor modifications, a WER of about 9% can be achieved. The modifications that can be employed are as follows:

-  Adding regularization to the loss function
- Pretraining the model up to 37 words (at 38 words, the PyTorch CTC loss function limit gets exceeded, however, using something like resampling if there's a bad sample, the model can be pretrained up to much more words)
- More finer curriculum learning steps
- Hyperparameter search
- Data Augmentation
- Making CTC loss deterministic so that the results can be reproduced

(To be done after getting some significant decrease in WER of the presently trained model)

## To Do

- Adding ignored files to github, unsetting git credentials
- deleting all wav and png files from dataset at last
- Predictions logging

## References

1. Afouras T. and Chung J. Deep Lip Reading: a comparison of models and an online application, 2018. [(GitHub)](https://github.com/afourast/deep_lip_reading)
2. Harald Scheidl. CTC Decoding Algorithms [(GitHub)](https://github.com/githubharald/CTCDecoder)