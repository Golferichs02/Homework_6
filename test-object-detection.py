""" test-object-detection.py
This script reads a video provided by the user alongside the resize percentage of the 
original video size frame and performs a colour space conversion from rgb to hsv to each frame,
then it filters the black and blue colors with the inrange and bitwise_or function from cv2 to the
current frame, next with the detected binary array it tracks within a 10 pixel radius from frame
to frame the non zero pixels a.k.a the white pixels so the coordinates can be implemented in the 
cv2.rectangel function to the original image so it tracks the person in the footage.

Authors: Emilio Arredondo Payán (628971) & Jorge Alberto Rosales de Golferichs (625544) 
Contacts: emilio.arredondop@udem.edu, jorge.rosalesd@udem.edu
Organisation: Universidad de Monterrey
First created on Tuesday 19 march 2024
"""

import cv2 
import numpy
import argparse as arg
import od 

def run_pipeline()->None:
    args = od.user_interaction()
    cap = od.initialise_camera(args)
    od.segment_object(cap, args)
    od.close_windows(cap)
    return None

if __name__ == "__main__":
    run_pipeline()