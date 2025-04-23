import numpy as np
import cv2
import utilities as ut

webcam = False # Set to True if using a webcam, False if using a picture or video file
path = '1.jpg'
cap = cv2.VideoCapture(0) 
cap.set(10, 160) # Brightness
cap.set(3, 1920) # Width
cap.set(4, 1080) # Height

while True:
    
    if webcam:success, img = cap.read()
    else: img = cv2.imread(path)

    ut.getContours(img,showCanny=True) # Get contours from the image

    img = cv2.resize(img, (0,0),None,0.5,0.5)
    cv2.imshow('Original',img)
    cv2.waitKey(1)
