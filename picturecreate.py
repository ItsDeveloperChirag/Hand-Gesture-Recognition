import time
import numpy as np
import math

import mediapipe
import cv2
from cvzone.HandTrackingModule import HandDetector
cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
offset=20
imgsize=300



folder="C:/Users/dell/Desktop/Handwritten text genration - Copy/closeChrome"

counter=0
while True:
    success,img=cap.read()
    hands,img=detector.findHands(img)

    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgwhite=np.ones((imgsize,imgsize,3),np.uint8)*255
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgCropshape=imgCrop.shape
        aspectratio=h/w

        if aspectratio>1:
            k=imgsize/h
            wcal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wcal,imgsize))
            imgResizeshape = imgResize.shape
            wGap=math.ceil((imgsize-wcal)/2)
            imgwhite[:,wGap:wcal+wGap] = imgResize


        else:
            k = imgsize / w
            hcal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgsize,hcal ))
            imgResizeshape = imgResize.shape
            hGap = math.ceil((imgsize - hcal) / 2)
            imgwhite[hGap:hcal + hGap, : ] = imgResize

        cv2.imshow('abcd',imgCrop)
        cv2.imshow('abc', imgwhite)


    cv2.imshow('asd',img)
    k=cv2.waitKey(1)
    if k==ord("s"):
        counter +=1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg',imgwhite)
        print(counter)





