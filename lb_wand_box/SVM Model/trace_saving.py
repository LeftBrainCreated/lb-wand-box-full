# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 22:51:23 2020

@author: John
"""

import numpy as np
import cv2
import os
import time
from picamera2 import Picamera2

# initialize the camera and configure
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

# allow the camera to warmup
time.sleep(1.0)

#%% Create Blob Detector
# Define parameters for the required blob
params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 160
params.maxThreshold = 255

params.filterByColor = 1
params.blobColor = 255

params.filterByCircularity = 1
params.minCircularity = 0.5

params.filterByConvexity = 1
params.minConvexity = 0.6

params.filterByArea = 1
params.minArea = 8
params.maxArea = 100

params.filterByInertia = 1
params.minInertiaRatio = 0.5

# creating object for SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)
# bg_mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=50, detectShadows=False)

## List to hold coordinates of detected blobs and hold list of blob points
blob_points = []

# Frame Size
w_frame = 480;
h_frame = 640;

# Image Resize 
scale_percent = 15 # percent of original size
width = int(w_frame * scale_percent / 100)
height = int(h_frame * scale_percent / 100)
dim = (height,width)

im_save_fp = r'/home/leftbrain/Desktop/Raspberry_Potter/Rb_potter_files/Pictures'
im_name = "random"
im_suffix = "%d.png"
print(os.path.join(im_save_fp, im_name))

im_name = im_name + im_suffix

count = 0
trace_len = 25
im_count = 0
im_count_limit = 50

# Create Blank Frame to Overlay key points
blank_image = np.zeros((w_frame,h_frame,3), np.uint8)

# capture frames from the camera in a loop
while True:
    # grab the frame as a NumPy array
    frame = picam2.capture_array()

    # fg_mask = bg_mog.apply(frame)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Turn to gray scale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # ret, thresh = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
    
    # # Displaying the output image
    # cv2.imshow('Binary Threshold', thresh)    
    
    # Detecting keypoints on video stream
    keypoints = detector.detect(frame)
    # mog_kp = detector.detect(fg_mask)
    
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # mog_with_keypoints = cv2.drawKeypoints(fg_mask, mog_kp, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size / 2)
        
        # Draw an outer circle to make keypoints more visible
        cv2.circle(frame_with_keypoints, (x, y), radius + 3, (255, 0, 0), 3)  # blue thick outline
        cv2.circle(frame_with_keypoints, (x, y), radius, (0, 255, 255), -1)  # filled yellow keypoint

    # cv2.imshow('Foreground Mask', mog_with_keypoints)

    # Get coordinates of blob
    points_array = cv2.KeyPoint_convert(keypoints)

    # # # # # # # # # # 
    # # testing
    # # # # # # # # # # 

    cv2.imshow("Keypoints", frame_with_keypoints);

    # # # # # # # # # # # # # # # # 
    
    # # Initialize Points array
    # if len(points_array) != 0:
    #     blob_points.append(points_array[0])
    
    # # Draw the path by drawing lines between 2 consecutive points in points list
    # for i in range(1, len(blob_points)):
    #     # Ensure the coordinates are integers
    #     pt1 = tuple(map(int, blob_points[i-1]))
    #     pt2 = tuple(map(int, blob_points[i]))
        
    #     cv2.line(blank_image, pt1, pt2, (255, 255, 255), 3)
    #     cv2.line(frame_with_keypoints, pt1, pt2, (255, 0, 0), 3)
        
    # # Show image with keypoints and trace  
    # cv2.imshow("frame", frame_with_keypoints)
        
    # # Truncate wand trace length and save image
    # if count > trace_len:
    #     # Start new trace 
    #     blob_points = []
        
    #     # Rescale Image for Saving
    #     resized = cv2.resize(blank_image, dim, interpolation=cv2.INTER_AREA)
        
    #     # Save Old trace
    #     cv2.imwrite(os.path.join(im_save_fp, im_name) % im_count, resized)
        
    #     # Indicator that image was saved
    #     cv2.circle(blank_image, (int(w_frame / 2), int(h_frame / 2)), 20, (0, 255, 0), 3)
        
    #     # Reset Blank Frame
    #     blank_image = np.zeros((w_frame, h_frame, 3), np.uint8)    
    
    #     # time.sleep(3.0)
        
    #     im_count += 1
    #     count = 0
        
    #     if im_count > im_count_limit:
    #         break 
    
    # count += 1
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()