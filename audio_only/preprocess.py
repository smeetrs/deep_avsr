import os
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np

from config import args
from utils.preprocessing import preprocess_sample



#walking through the data directory and obtaining a list of all files in the dataset
filesList = list()
for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
    for file in files:
        if file.endswith(".mp4"):
            filesList.append(os.path.join(root, file[:-4]))


#Preprocessing each sample
print("\nNumber of data samples to be processed = %d" %(len(filesList)))
print("\n\nStarting preprocessing ....\n")

for file in tqdm(filesList):
    preprocess_sample(file)

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

print("\nNoise file generated.\n")