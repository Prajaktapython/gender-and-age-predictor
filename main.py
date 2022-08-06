import cv2
import math
import argparse

from cv2 import FONT_HERSHEY_SIMPLEX
from cv2 import LINE_AA
def highlightface (net,frame,conf_threshold=0.7):
    frameopencvdnn = frame.copy()
    frameheight=frameopencvdnn.shape[0]
    framewidth=frameopencvdnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameopencvdnn,1.0,(300,300),[104,117,123],True,False)
    net.setInput(blob)
    detection=net.forward()
    facebox = []
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detection[0,0,i,3]* framewidth)
            y1=int(detection[0,0,i,4]* frameheight)
            x2=int(detection[0,0,i,5]* framewidth)
            y2=int(detection[0,0,i,6]*frameheight)
            facebox.append([x1,y1,x2,y2])
            cv2.rectangle(frameopencvdnn,(x1,y1),(x2,y2),(0,255,0),int(round(frameheight/150)),8)
    return frameopencvdnn,facebox        

parser = argparse.ArgumentParser()
parser.add_argument('--image')
args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
video=cv2.VideoCapture(args.image if args.image else 0)
padding = 20
while cv2.waitKey(1)<0 :
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    resultimg,faceboxes = highlightface(faceNet,frame)
    if not faceboxes:
        print ("no face detected")
    for facebox in faceboxes:
        face=frame[max(0,facebox [1]-padding):min(facebox[3]+padding,frame.shape[0]-1),
                   max(0,facebox[0]-padding):min(facebox[2]+padding,frame.shape [1]-1)] 
        blob =cv2.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False) 
        genderNet.setInput(blob) 
        genderpredict=genderNet.forward()
        gender=genderList[genderpredict[0].argmax()]
        print (f"gender:{gender}")

        ageNet.setInput(blob) 
        agepredict=ageNet.forward()
        age=ageList[agepredict[0].argmax()]
        print (f"age:{age[1:-1]}years")
        cv2.putText(resultimg,f'{gender},{age}',(facebox[0],facebox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)


        cv2.imshow('detect age and gender',resultimg)