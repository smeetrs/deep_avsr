# Deep Audio-Visual Speech Recognition

The repository contains a PyTorch reproduction of the [Deep Audio-Visual Speech Recognition](https://arxiv.org/abs/1809.02108) paper. We train three models - Audio-Only (AO), Video-Only (VO) and Audio-Visual (AV), on the [LRS2 dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) for the speech-to-text transcription task.

## Requirements

System packages:

	ffmpeg==2.8.15

Python packages:

	editdistance==0.5.3
	matplotlib==3.1.1
	numpy==1.18.1
	opencv-python==4.2.0
	pytorch==1.2.0
	scipy==1.3.1
	tqdm==4.42.1

CUDA 10.0 (if NVIDIA GPU is to be used):

	cudatoolkit==10.0


## Project Structure

The structure of the `audio_only`, `video_only` and `audio_visual` directories is as follows:

#### Directories

`/checkpoints`: Temporary directory to store intermediate model weights and plots while training. Gets automatically created.

`/data`: Directory containing the LRS2 Main and Pretrain dataset class definitions and other required data-related utility functions.

`/final`: Directory to store the final trained model weights and plots. If available, place the pre-trained model weights in the `models` subdirectory.

`/models`: Directory containing the class definitions for the models.

`/utils`: Directory containing function definitions for calculating CER/WER, greedy search/beam search decoders and preprocessing of data samples. Also contains functions to train and evaluate the model.

#### Files

`checker.py`: File containing checker/debug functions for testing all the modules and the functions in the project as well as any other checks to be performed.

`config.py`: File to set the configuration options and hyperparameter values.

`demo.py`: Python script for generating predictions with the specified trained model for all the data samples in the specified demo directory.

`preprocess.py`: Python script for preprocessing all the data samples in the dataset.

 `pretrain.py`: Python script for pretraining the model on the pretrain set of the LRS2 dataset using curriculum learning.

`test.py`: Python script to test the trained model on the test set of the LRS2 dataset.

`train.py`: Python script to train the model on the train set of the LRS2 dataset.


## Results

We provide Word Error Rate (WER) achieved by the models on the test set of the LRS2 dataset with both Greedy Search and Beam Search (with Language Model) decoding techniques. We have tested in cases of clean audio and noisy audio (0 dB SNR). We also give WER in cases where only one of the modalities is used in the Audio-Visual model.

<table>
<thead>
  <tr>
    <th>Operation Mode</th>
    <th colspan="2">AO/VO Model</th>
    <th colspan="2">AV Model</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td>Greedy</td>
    <td>Beam (+LM)<br></td>
    <td>Greedy</td>
    <td>Beam (+LM)</td>
  </tr>
  <tr>
    <td colspan="5">Clean Audio</td>
  </tr>
  <tr>
    <td>AO</td>
    <td>11.4%</td>
    <td>8.3%</td>
    <td>12.0%</td>
    <td>8.2%</td>
  </tr>
  <tr>
    <td>VO</td>
    <td>61.8%</td>
    <td>55.3%</td>
    <td>56.3%</td>
    <td>49.2%</td>
  </tr>
  <tr>
    <td>AV</td>
    <td>-</td>
    <td>-</td>
    <td>10.3%</td>
    <td>6.8%</td>
  </tr>
  <tr>
    <td colspan="5">Noisy Audio</td>
  </tr>
  <tr>
    <td>AO</td>
    <td>62.5%</td>
    <td>54.0%</td>
    <td>59.0%</td>
    <td>50.7%</td>
  </tr>
  <tr>
    <td>AV</td>
    <td>-</td>
    <td>-</td>
    <td>29.1%</td>
    <td>22.1%</td>
  </tr>
</tbody>
</table>


## Pre-trained Weights

Download the pre-trained weights for the Visual Frontend using the following command:

	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1k0Zk90ASft89-xAEUbu5CmZWih_u_lRN' -O visual_frontend.pt

Download the pre-trained weights for the Language Model using the following command:

	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Gc1YqaTCEqrITYOWPBvmLezE6Fv_e4fu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Gc1YqaTCEqrITYOWPBvmLezE6Fv_e4fu" -O language_model.pt && rm -rf /tmp/cookies.txt

Once the Visual Frontend and Language Model weights are downloaded, place them in a folder and add their paths in the `config.py` file.

The pre-trained weights of the AO, VO and AV models will be made available soon.

[//]: # (Please send an email at `smeetrs<AT>gmail.com` from your institutional email ID for the pre-trained weights of the AO, VO and AV models. Place the weights of each model in the corresponding `/final/models` directory.)


## How To Use

If planning to train the models, download the complete LRS2 dataset from [here](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) or in cases of custom datasets, have the specifications and folder structure similar to LRS2 dataset.

Steps have been provided to either train the models or to use the trained models directly for inference:

#### Training

Set the configuration options in the `config.py` file before each of the following steps as required. Comments have been provided for each option.

1. Run the `preprocess.py` script to preprocess and generate the required files for each sample.

2. Run the `pretrain.py` script for one iteration of curriculum learning. Run it multiple times, each time changing the `PRETRAIN_NUM_WORDS` argument in the `config.py` file to perform multiple iterations of curriculum learning.

3. Run the `train.py` script to finally train the model on the train set.

4. Once the model is trained, run the `test.py` script to obtain the performance of the trained model on the test set.

5. Run the `demo.py` script to use the model to make predictions for each sample in a demo directory. Read the specifications for the sample in the `demo.py` file.

#### Inference

1. Set the configuration options in the `config.py` file. Comments have been provided for each option.

2. Run the `demo.py` script to use the model to make predictions for each sample in a demo directory. Read the specifications for the sample in the `demo.py` file.


## References

1. The pre-trained weights of the Visual Frontend and the Language Model have been obtained from [Afouras T. and Chung J, Deep Lip Reading: a comparison of models and an online application, 2018](https://github.com/afourast/deep_lip_reading) GitHub repository.

2. The CTC beam search implementation is adapted from [Harald Scheidl, CTC Decoding Algorithms](https://github.com/githubharald/CTCDecoder) GitHub repository.

***

*PS: Please do not hesitate to raise an issue in case of any bugs/doubts/suggestions. Happy Open Source-ing !!* 😃
