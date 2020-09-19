"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import os



def preprocess_sample(file):
    """
    Function to preprocess each data sample.
    Extracts the audio from the video file using the FFmpeg utility and saves it to a wav file.
    """
    videoFile = file + ".mp4"
    audioFile = file + ".wav"
    v2aCommand = "ffmpeg -y -v quiet -i " + videoFile + " -ac 1 -ar 16000 -vn " + audioFile
    os.system(v2aCommand)
    return
