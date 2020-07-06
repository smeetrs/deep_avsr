import torch
from scipy.io import wavfile
from tqdm import tqdm
import numpy as np
import os

from config import args
from models.visual_frontend import VisualFrontend
from utils.preprocessing import preprocess_sample



np.random.seed(args["SEED"])
torch.manual_seed(args["SEED"])
gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")



#declaring the visual frontend module
vf = VisualFrontend()
vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"], map_location=device))
vf.to(device)


#walking through the data directory and obtaining a list of all files in the dataset
filesList = list()
for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
    for file in files:
        if file.endswith(".mp4"):
            filesList.append(os.path.join(root, file[:-4]))


#Preprocessing each sample
print("\nNumber of data samples to be processed = %d" %(len(filesList)))
print("\n\nStarting preprocessing ....\n")

params = {"roiSize":args["ROI_SIZE"], "normMean":args["NORMALIZATION_MEAN"], "normStd":args["NORMALIZATION_STD"], "vf":vf}
for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
    preprocess_sample(file, params)

print("\nPreprocessing Done.")



#Generating a 1 hour noise file
#Fetching audio samples from 20 random files in the dataset and adding them up to generate noise
#The length of these clips is the shortest audio sample among the 20 samples
print("\n\nGenerating the noise file ....")

noise = np.empty((0))
while len(noise) < 16000*3600:
    noisePart = np.zeros(16000*60)
    indices = np.random.randint(0, len(filesList), 20)
    for ix in indices:
        sampFreq, audio = wavfile.read(filesList[ix] + ".wav")
        audio = audio/np.max(np.abs(audio))
        pos = np.random.randint(0, abs(len(audio)-len(noisePart))+1)
        if len(audio) > len(noisePart):
            noisePart = noisePart + audio[pos:pos+len(noisePart)]
        else:
            noisePart = noisePart[pos:pos+len(audio)] + audio
    noise = np.concatenate([noise, noisePart], axis=0)
noise = noise[:16000*3600]
noise = (noise/20)*32767
noise = np.floor(noise).astype(np.int16)
wavfile.write(args["DATA_DIRECTORY"] + "/noise.wav", 16000, noise)

print("\nNoise file generated.")



#Generating preval.txt for splitting the pretrain set into train and validation sets
print("\n\nGenerating the preval.txt file ....")

with open(args["DATA_DIRECTORY"] + "/pretrain.txt", "r") as f:
    lines = f.readlines()

if os.path.exists(args["DATA_DIRECTORY"] + "/preval.txt"):
    with open(args["DATA_DIRECTORY"] + "/preval.txt", "r") as f:
        lines.extend(f.readlines())

indices = np.arange(len(lines))
np.random.shuffle(indices)
valIxs = np.sort(indices[:int(np.ceil(args["PRETRAIN_VAL_SPLIT"]*len(indices)))])
trainIxs = np.sort(indices[int(np.ceil(args["PRETRAIN_VAL_SPLIT"]*len(indices))):])

lines = np.sort(np.array(lines))
with open(args["DATA_DIRECTORY"] + "/pretrain.txt", "w") as f:
    f.writelines(list(lines[trainIxs]))
with open(args["DATA_DIRECTORY"] + "/preval.txt", "w") as f:
    f.writelines(list(lines[valIxs]))

print("\npreval.txt file generated.\n")
