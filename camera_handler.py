#!/usr/bin/python

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

from lane_detection import Lane_Detection

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (1280, 720)
camera.framerate = 5

rawCapture = PiRGBArray(camera, size=(1280, 720))

# allow the camera to warmup
time.sleep(0.1)

#create instance of class Lane_Detection
ld = Lane_Detection()

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    
    #load the new image into ld
    ld.load_image(image)
    
    #run the algorithm
    ld.calc_binary()
    #ld.warp_image()
    #ld.find_lane_pixels()
    #ld.fit_polynomial()

    # show the frame
    cv2.imshow("Frame", ld.temp_img)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
