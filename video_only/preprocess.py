import torch
import os
from tqdm import tqdm

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
            