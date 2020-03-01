import torch
import os
from tqdm import tqdm

from config import args
from models.visual_frontend import VisualFrontend
from utils.preprocessing import preprocess_sample



filesList = list()
for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
    for file in files:
        if file.endswith(".mp4"):
            filesList.append(os.path.join(root, file[:-4]))


print("\nNumber of data samples to be processed = %d\n" %(len(filesList)))
print("\nStarting preprocessing ....\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vf = VisualFrontend().to(device)
vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"]))
vf.to(device)
vf.eval()
params = {"roiSize":args["ROI_SIZE"], "normMean":args["NORMALIZATION_MEAN"], "normStd":args["NORMALIZATION_STD"], "vf":vf}

for file in tqdm(filesList):
    preprocess_sample(file, params)
            
print("\nPreprocessing Done.\n")
            