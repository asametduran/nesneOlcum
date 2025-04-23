import numpy as np
import cv2

webcam = False # Set to True if using a webcam, False if using a picture or video file
path = '1.jpg'
cap = cv2.VideoCapture(0) 
cap.set(10, 160) # Brightness
cap.set(3, 1920) # Width
cap.set(4, 1080) # Height

while True:
    success, img = cap.read()
    
    cv2.imshow('Original',img)
    cv2.waitKey(1)
