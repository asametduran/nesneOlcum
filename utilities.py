import cv2
import numpy as np

def getContours(img,cannyThreshHold=[100,100],showCanny=False, minArea=1000,filter=0, draw=False):
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
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            boundingBox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx), area, approx, boundingBox, i])
            else:
                finalContours.append([len(approx), area, approx, boundingBox, i])

    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True) # Sort contours by area, largest first

    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

    return img, finalContours

def reorder(myPoints):
    #print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints) # Create a new array with the same shape as myPoints
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1) # Sum of x and y coordinates
    myPointsNew[0] = myPoints[np.argmin(add)] # Top left point
    myPointsNew[3] = myPoints[np.argmax(add)] # Bottom right point
    diff = np.diff(myPoints, axis=1) # Difference between x and y coordinates
    myPointsNew[1] = myPoints[np.argmin(diff)] # Top right point
    myPointsNew[2] = myPoints[np.argmax(diff)] # Bottom left point
    return myPointsNew

def warpImg(img, points, w, h, pad=20):
  
    #print(points) #[[ 891  187]]
    # [[ 146  201]]
    #
    # [[ 117 1669]]
    #
    # [[ 934 1667]]]
    #   (4, 1, 2)

    points = reorder(points)

    pts1 = np.float32(points) # Convert points to float32
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) # Destination points
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # Get perspective transform matrix
    imgWarp = cv2.warpPerspective(img, matrix, (w, h)) # Warp the image using the matrix
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad] # Crop the image to remove padding [h,w]

    return imgWarp

def findDistance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))
