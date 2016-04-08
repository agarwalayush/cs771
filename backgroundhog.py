import sys
import cv2
from skimage.feature import hog
import numpy as np
from config import *
from sklearn.externals import joblib
from skimage.transform import resize
import pickle

filename = sys.argv[1]

kernelo = np.ones((10,10),np.uint8)
kernelc = np.ones((5,5),np.uint8)
kerneld = np.ones((6,6),np.uint8)

cap = cv2.VideoCapture(filename)
red =3
history = 500
print("hello")
svm = joblib.load(model_path)
#m = pickle.load(open("clustering/object500.p","rb"))
fgbg = cv2.createBackgroundSubtractorMOG2(history = history, varThreshold=16, detectShadows = False)
WIDTH = 100
HEIGHT = WIDTH
# fgbg.setShadowValue(0)
frameNo = 1
print("start")
while(1):
    ret, frame = cap.read()
    if ret == False:
        break
    height = len(frame)
    width = len(frame[0])

    frame = cv2.GaussianBlur(frame, (21, 21), 0)
    fgmask = fgbg.apply(frame, learningRate = 1.0/history)
    if(frameNo%25 == 0):
        fgmask = cv2.resize(fgmask, (int(width/red),int(height/red)))

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernelo)
        # fgmask = cv2.dilate(fgmask, kerneld, iterations = 4)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernelc)

        # fgmask = cv2.resize(fgmask, (width/2,height/2))
        (_,cnts, _) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# loop over the contours
        frame = cv2.resize(frame, (int(width/red),int(height/red)))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        for c in cnts:
            if cv2.contourArea(c) < 700:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(fgmask, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # sift = cv2.SIFT()
            im = frame[(y/red):((y+h)/red),(x/red):((x+w)/red)]
            if(min(im.shape) != 0):
                im = resize(im,(WIDTH, HEIGHT) ,mode='nearest')
               # frame = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
                # if(min(im.shape) != 0):
                #     key1, des1 = sift.detectAndCompute(im, None)
                #     if des1 != None:
                #         a = [0 for i in range(500)]
                #         for i in range(len(des1)):
                #                 a[m.predict(des1)[i]]+=1
                font = cv2.FONT_HERSHEY_SIMPLEX
                s = str(svm.predict(fd)[0])
                cv2.putText(frame,s,(x,y), font, 1,(255,255,255),2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = "Occupied"

        cv2.imshow('frame',frame)
        # cv2.waitKey(1000)
        # cv2.imshow('frame',fgmask)
        cv2.waitKey(1)

    frameNo += 1
cap.release()
cv2.destroyAllWindows()
