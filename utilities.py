import cv2
import numpy as np

def getContours(img,cannyThreshHold=[100,100],showCanny=False, minArea=1000,filter=0):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # Apply Gaussian blur
    imgCanny = cv2.Canny(imgBlur, cannyThreshHold[0],cannyThreshHold[1]) # Apply Canny edge detection
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3) # Dilate the image
    imgThre = cv2.erode(imgDial, kernel, iterations=2) # Erode the image
    if showCanny:cv2.imshow('Canny',imgThre) # Show Canny image

    contours, hierarchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours
    finalContours = [] # List to store final contours
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True) # Get perimeter, closed
            approx = cv2.approxPolyDP(i, 0.02 * peri, True) # Approximate the contour to a polygon
            boundingBox = cv2.boundingRect(approx) # Get bounding box
            if filter > 0 :
                if len(approx) == filter:
                    finalContours.append([len(approx),area, approx, boundingBox, i])
                else:
                    finalContours.append([len(approx),area, approx, boundingBox, i])


