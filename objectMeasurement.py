import numpy as np
import cv2
import utilities as ut

webcam = False # Set to True if using a webcam, False if using a picture or video file
path = '2.jpg'
cap = cv2.VideoCapture(0) 
cap.set(10, 160) # Brightness
cap.set(3, 1920) # Width
cap.set(4, 1080) # Height
scale = 3 # Scale factor for the image
wP = 210 *scale # Width of the paper in cm
hP = 297*scale # Height of the paper in cm

while True:
    
    if webcam:success, img = cap.read()
    else: img = cv2.imread(path)

    finalImg, finalContours = ut.getContours(img,
                                            minArea=50000,filter=4) # Get contours from the image
    
    if len(finalContours) != 0:
        biggestContour = finalContours[0][2]
        #print(biggestContour) #u çıkıtyı alıyorsun 891,197   146,201     117,1669     934 1667       4 1 2
        imgWarp = ut.warpImg(finalImg, biggestContour, wP, hP) # Warp the image using the biggest contour
        cv2.imshow('A4',imgWarp)

    img = cv2.resize(img, (0,0),None,0.5,0.5)
    cv2.imshow('Original',img)
    cv2.waitKey(1)
