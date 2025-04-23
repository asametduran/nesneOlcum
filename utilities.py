import cv2
import numpy as np

def getContours(img,cannyThreshHold=[100,100],showCanny=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # Apply Gaussian blur
    imgCanny = cv2.Canny(imgBlur, cannyThreshHold[0],cannyThreshHold[1]) # Apply Canny edge detection
    if showCanny:cv2.imshow('Canny',imgCanny) # Show Canny image