# import required packages
import cv2 as cv
import dlib
import argparse
import numpy as np
import os
from scipy import signal
from scipy.io import wavfile




#parser to obtain command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", default="./data", help="Path to data")
parser.add_argument("--facemodel", default="./pretrained/mmod_human_face_detector.dat", 
                    help="Path to pre-trained CNN face detection model")
parser.add_argument("--lm68model", default="./pretrained/shape_predictor_68_face_landmarks.dat",
                    help="Path to pre-trained dlib 68-point landmark detector")
parser.add_argument("--visualize", default=False, help="Displays the video with face and mouth bounding boxes")
args = parser.parse_args()




#function to convert dlib shape object to a numpy array
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords





#walk through the data directory and obtain a list of all the mp4 files
videoFilesList = list()
for root, dirs, files in os.walk(args.datadir):
    for file in files:
        if file.endswith(".mp4"):
            videoFilesList.append(os.path.join(root, file))





for file in videoFilesList:


    #extract video stream and loop through
    videoFile = cv.VideoCapture(file)

    #creating face and landmarks detector objects
    cnnFaceDetector = dlib.cnn_face_detection_model_v1(args.facemodel)
    landmarkDetector = dlib.shape_predictor(args.lm68model)

    #array to store the top left coordinates of the mouth ROI of each frame
    mouthTopLeftCoords = np.empty((0,2), dtype=np.int)



    while (videoFile.isOpened()):
        
        ret, frame = videoFile.read()

        if ret == True:
            
            #resize the frame to 224x224 and convert it to grayscale
            frame = cv.resize(frame, (224,224), interpolation=cv.INTER_CUBIC)
            grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


            #run the cnn face detector to obtain the faces in the frame
            faces = cnnFaceDetector(grayed, 0)

            #if faces are found
            if len(faces) != 0:

                #if multiple faces found, choose the face with largest bounding box as the speaker face
                if len(faces) > 1:
                    maxArea = 0
                    for i in range(len(faces)):
                        x = faces[i].rect.left()
                        y = faces[i].rect.top()
                        w = faces[i].rect.right() - x
                        h = faces[i].rect.bottom() - y
                        area = w*h
                        if area > maxArea:
                            maxArea = area
                            speakerFace = faces[i].rect

                #if only one face found, choose the face to be the speaker face
                else:
                    speakerFace = faces[0].rect

                #draw a bounding box around the speaker face if visualize is set to True
                if args.visualize:
                    cv.rectangle(frame, (speakerFace.left(), speakerFace.top()), (speakerFace.right(), speakerFace.bottom()), (0,0,255), 1)



                #find the 68 landmark points on the speaker's face
                landmarkPts = landmarkDetector(grayed, speakerFace)
                landmarkPts = shape_to_np(landmarkPts) 

                #get the landmark points corresponding to the mouth region           
                mouthPts = landmarkPts[60:68,:]

                #get the top left coordinates of the mouth ROI which is centered at the mouth
                mouthTopLeft = np.round(np.mean(mouthPts, axis=0, keepdims=True) - 56).astype(np.int)
                
                #draw the mouth ROI if visualize is set to True
                if args.visualize:
                    cv.rectangle(frame, (mouthTopLeft[0,0], mouthTopLeft[0,1]), (mouthTopLeft[0,0]+112, mouthTopLeft[0,1]+112), (255,0,0), 2)



            #if no face is found, insert (-1,-1) to indicate no faces found
            else:
                mouthTopLeft = np.array([[-1, -1]])


            #stack the top left coordinates in the array
            mouthTopLeftCoords = np.vstack((mouthTopLeftCoords, mouthTopLeft))

            #display the video
            if args.visualize:
                cv.imshow("Mouth Region Extraction", frame)
                if cv.waitKey(1) == ord('q'):
                   break


        else:
            break


    videoFile.release()
    cv.destroyAllWindows()


    #runs a linear interpolation algorithm to the get coordinates for the frames where no face was found
    #a linear interpolation is done using the coordinates of the two closest frames (one past frame, other future frame)

    videoLen = mouthTopLeftCoords.shape[0]

    #if first few frames have missing coordinates, copy the first non-missing coordinates in all these
    frameNum = 0
    if mouthTopLeftCoords[frameNum,0] == -1 or mouthTopLeftCoords[frameNum,1] == -1:      
                
        fn = frameNum + 1
        while (mouthTopLeftCoords[fn,0] == -1 or mouthTopLeftCoords[fn,1] == -1):
            fn = fn + 1
        tempMouthTL = mouthTopLeftCoords[fn,:]

        fn = frameNum
        while (mouthTopLeftCoords[fn,0] == -1 or mouthTopLeftCoords[fn,1] == -1):
            mouthTopLeftCoords[fn,:] = tempMouthTL
            fn = fn + 1



    #if the last few frames have missing coordinates, copy the last non-missing coordinates in all these
    frameNum = videoLen-1
    if mouthTopLeftCoords[frameNum,0] == -1 or mouthTopLeftCoords[frameNum,1] == -1: 
        
        fn = frameNum - 1
        while (mouthTopLeftCoords[fn,0] == -1 or mouthTopLeftCoords[fn,1] == -1):
            fn = fn - 1
        tempMouthTL = mouthTopLeftCoords[fn,:]

        fn = frameNum
        while (mouthTopLeftCoords[fn,0] == -1 or mouthTopLeftCoords[fn,1] == -1):
            mouthTopLeftCoords[fn,:] = tempMouthTL
            fn = fn - 1



    #if intermediate few frames have missing coordinates, linearly interpolate using 
    #the previously seen and the next seen non-missing coordinates
    for frameNum in range(videoLen):
        
        if mouthTopLeftCoords[frameNum,0] == -1 or mouthTopLeftCoords[frameNum,1] == -1:      

            initMouthTL = mouthTopLeftCoords[frameNum-1,:]
            fn = frameNum+1
            while (mouthTopLeftCoords[fn,0] == -1 or mouthTopLeftCoords[fn,1] == -1):
                fn = fn + 1
            finalMouthTL = mouthTopLeftCoords[fn,:]
            
            numPoints = fn - frameNum + 2
            xValues = np.round(np.linspace(initMouthTL[0], finalMouthTL[0], numPoints)).astype(np.int)
            yValues = np.round(np.linspace(initMouthTL[1], finalMouthTL[1], numPoints)).astype(np.int)

            fn = frameNum
            while (mouthTopLeftCoords[fn,0] == -1 or mouthTopLeftCoords[fn,1] == -1):
                mouthTopLeftCoords[fn,:] = [xValues[fn - frameNum + 1], yValues[fn - frameNum + 1]]
                fn = fn + 1


    #save the mouth ROI top left coordinates in an mtlc224 txt file
    np.savetxt(file[:-4] + '_mtlc224.txt', mouthTopLeftCoords, fmt="%d")



    #run the video again, extract the mouth ROI image in all frames and stack them into one single image
    videoFile = cv.VideoCapture(file)
    mouthPatchSequence = np.empty((112,0), dtype=np.int)
    frameNum = 0

    while (videoFile.isOpened()):
        
        ret, frame = videoFile.read()

        if ret == True:
            
            frame = cv.resize(frame, (224,224), interpolation=cv.INTER_CUBIC)
            grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            mouthTopLeft = mouthTopLeftCoords[frameNum, :]
            mouthROI = grayed[mouthTopLeft[1]:mouthTopLeft[1]+112, mouthTopLeft[0]:mouthTopLeft[0]+112]

            mouthPatchSequence = np.hstack((mouthPatchSequence, mouthROI))

        else:
            break

        frameNum = frameNum + 1

    videoFile.release()

    #save the mouth ROI for all frames image in an roi png file
    cv.imwrite(file[:-4] + '_roi.png', mouthPatchSequence)




    #code to extract the stft of the audio 

    #extract the audio from the video file and store it in a wav file
    v2aCommand = 'ffmpeg -y -v quiet -i ' + file + ' -ac 1 -ar 16000 -vn ' + file[:-4] + '.wav'
    os.system(v2aCommand)

    #read the audio file and compute its stft
    sampFreq, audioFile = wavfile.read(file[:-4] + '.wav')
    freqs, time, stftVals = signal.stft(audioFile, sampFreq, window='hamming', nperseg=640, noverlap=480, boundary=None, padded=False)

    #remove the extra last samples so that the number of stft samples equals 4 times the number of frames in video
    stftVals = stftVals[:,:videoLen*4]

    #take the magnitude
    stftMag = np.abs(stftVals)

    #save the stft values in an stft txt file
    np.savetxt(file[:-4] + '_stft.txt', stftMag, fmt="%.6e")