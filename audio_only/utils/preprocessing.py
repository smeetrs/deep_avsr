import os


def preprocess_sample(file):
    videoFile = file + ".mp4"
    audioFile = file + ".wav"
    v2aCommand = "ffmpeg -y -v quiet -i " + videoFile + " -ac 1 -ar 16000 -vn " + audioFile
    os.system(v2aCommand)
    return