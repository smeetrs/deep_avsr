import torch
from scipy.io import wavfile
from tqdm import tqdm
import numpy as np
import os

from config import args
from models.visual_frontend import VisualFrontend
from utils.preprocessing import preprocess_sample



#declaring the visual frontend module
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
print("\nNumber of data samples to be processed = %d\n" %(len(filesList)))
print("\nStarting preprocessing ....\n")

params = {"roiSize":args["ROI_SIZE"], "normMean":args["NORMALIZATION_MEAN"], "normStd":args["NORMALIZATION_STD"], "vf":vf}
for file in tqdm(filesList):
    preprocess_sample(file, params)

print("\nPreprocessing Done.\n")
            
            

#Generating a 1 hour noise file
#Fetching audio samples from 20 random files in the dataset and adding them up to generate noise
#The length of these clips is the shortest audio sample among the 20 samples
print("\nGenerating the noise file ....\n")

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

print("Noise file generated.\n")
