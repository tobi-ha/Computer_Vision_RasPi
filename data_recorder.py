#!/usr/bin/python

import RPi.GPIO as GPIO
import time
from picamera import PiCamera
import datetime  # new

class Data_Recorder:
    is_initialized = False
    save_directory = "/home/pi/Documents/Projects/picam/recordings/"
    
    def __init__(self):
        self.is_initialized = True
        
    def record_gpio_4(self, gpio_pin):
        