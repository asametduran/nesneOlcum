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
        
        finalImg2, finalContours2 = ut.getContours(
            imgWarp,
            cannyThreshHold=[50, 50],
            minArea=2000,
            filter=4,
            draw=False # Get contours from the warped image
        )
        for obj in finalContours2:
                cv2.polylines(finalImg2,[obj[2]],True,(0,255,0),2)
                nPoints = ut.reorder(obj[2])
                nW = round((ut.findDistance(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)
                nH = round((ut.findDistance(nPoints[0][0]//scale,nPoints[2][0]//scale)/10),1)
                cv2.arrowedLine(finalImg2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(finalImg2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(finalImg2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(finalImg2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)

        cv2.imshow('A4', finalImg2) # Show the warped image with contours


    img = cv2.resize(img, (0,0),None,0.5,0.5)
    cv2.imshow('Original',img)
    cv2.waitKey(1)
