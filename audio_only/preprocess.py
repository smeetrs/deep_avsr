import numpy as np
import os

from config import args


for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
    for file in files:
        if file.endswith(".mp4"):
            videoFile = os.path.join(root, file)
            audioFile = os.path.join(root, file[:-4]) + ".wav"
            v2aCommand = "ffmpeg -y -v quiet -i " + videoFile + " -ac 1 -ar 16000 -vn " + audioFile
            os.system(v2aCommand)