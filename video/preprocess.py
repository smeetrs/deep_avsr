import numpy as np
import cv2 as cv
import os

from config import args


roiSize = args["ROI_SIZE"]
for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
    for file in files:
        if file.endswith(".mp4"):
            videoFile = os.path.join(root, file)
            roiFile = os.path.join(root, file[:-4]) + ".png"
            
            captureObj = cv.VideoCapture(videoFile)
            roiSequence = np.empty((roiSize,0), dtype=np.int)
            while (captureObj.isOpened()):
                ret, frame = captureObj.read()
                if ret == True:
                    frame = cv.resize(frame, (224,224), interpolation=cv.INTER_CUBIC)
                    grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    roi = grayed[int(112-(roiSize/2)):int(112+(roiSize/2)), int(112-(roiSize/2)):int(112+(roiSize/2))]
                    roiSequence = np.hstack((roiSequence, roi))
                else:
                    break

            captureObj.release()
            cv.imwrite(roiFile, roiSequence)