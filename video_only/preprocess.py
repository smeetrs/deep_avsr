import torch
import numpy as np
import cv2 as cv
import os

from config import args
from models.visual_frontend import VisualFrontend


roiSize = args["ROI_SIZE"]
normMean = args["NORMALIZATION_MEAN"]
normStd = args["NORMALIZATION_STD"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vf = VisualFrontend().to(device)
vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"]))
vf.to(device)
vf.eval()

for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
    for file in files:
        if file.endswith(".mp4"):
            videoFile = os.path.join(root, file)
            roiFile = os.path.join(root, file[:-4]) + ".png"
            visualFeaturesFile = os.path.join(root, file[:-4]) + ".npy"
            
            captureObj = cv.VideoCapture(videoFile)
            roiSequence = list()
            while (captureObj.isOpened()):
                ret, frame = captureObj.read()
                if ret == True:
                    grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    grayed = grayed/255
                    grayed = cv.resize(grayed, (224,224))
                    roi = grayed[int(112-(roiSize/2)):int(112+(roiSize/2)), int(112-(roiSize/2)):int(112+(roiSize/2))]
                    roiSequence.append(roi)
                else:
                    break

            captureObj.release()
            cv.imwrite(roiFile, np.floor(255*np.hstack(roiSequence)).astype(np.int))
            
            
            roiSequence = [roi.reshape((roi.shape[0],roi.shape[1],1)) for roi in roiSequence]
            inp = np.dstack(roiSequence)
            inp = np.transpose(inp, (2,0,1))
            inp = inp.reshape((inp.shape[0],1,1,inp.shape[1],inp.shape[2]))
            inp = (inp - normMean)/normStd
            inputBatch = torch.from_numpy(inp)
            inputBatch = (inputBatch.float()).to(device)
            with torch.no_grad():
                outputBatch = vf(inputBatch)
            out = outputBatch.view(outputBatch.size(0), outputBatch.size(2))
            out = out.cpu().numpy()
            np.save(visualFeaturesFile, out)
            
            