from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def faceMaskDetector(frame, faceDetectorModel,maskDetectorModel):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (148, 148),(104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceDetectorModel.setInput(blob)
    detections = faceDetectorModel.forward()

    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (148, 148))
            face = img_to_array(face)
            #face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))            
            
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskDetectorModel.predict(faces)
        
    return (locs,preds)


def plotSquare(frame, box, color):    
    (startX, startY, endX, endY) = box
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

def displayText(displayContent, textX, textY, color):
    cv2.putText(frame, displayContent, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

prototxtPath = r"faceDetector\deploy.prototxt"
weightsPath = r"faceDetector\res10_300x300_ssd_iter_140000.caffemodel"

faceDetectorModel = cv2.dnn.readNet(prototxtPath, weightsPath)
maskDetectorModel = load_model("maskDetectorCNN.model")


print("Turning on the camera..")
vs = VideoStream(src=0).start()


while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=700)
    (locs,preds) = faceMaskDetector(frame, faceDetectorModel,maskDetectorModel)
    
    #print(preds)
    
    for (box,pred) in zip(locs,preds):
        if np.round(pred) == 0:
            color = (0, 255, 0)
            conf = "Mask Detected: "+str(np.round(1-pred,3))
        else:
            color = (0, 0, 255)
            conf = "No Mask: "+str(np.round(pred,3))
        plotSquare(frame, box, color)
        displayText(conf, box[0], box[1] - 10, color)

    
    # Displaying the video as frames
    cv2.imshow("Face-mask Detector", frame)
    # Pressing a wait key to keep the session running
    key = cv2.waitKey(1) 

    # If we press the "x" key from the keyboard, the session ends
    if key == ord("x"):
            break
                
# Preparing to close all windows and quit the video stream
cv2.destroyAllWindows()
vs.stop()