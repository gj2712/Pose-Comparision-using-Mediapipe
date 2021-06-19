import cv2
import mediapipe as mp
import time
import math
import imutils
import Module_pose as pd
import numpy as np
import pickle
from numpy import dot
from numpy.linalg import norm
from scipy import spatial


class generate_vector():
    def __init__(self,path):
        self.path=path

    def form_pikle(self,output_name):

        cap = cv2.VideoCapture(self.path)
        if (cap.isOpened() == False):
            print("Error opening video  file")
        detector=pd.pose_detector()
        self.k=0
        my_list=[]
        while cap.isOpened():
            success, img = cap.read()
            if success:
                img = detector.find_body(img)
                bbox, pos_list, y_list, roi_co = detector.find_position(cv2.resize(img, (372, 495)))
                pos_list.extend(y_list)
                input_points = np.array(pos_list)
                toi = detector.roi(input_points, roi_co)
                input_new_coords = np.asarray(toi).reshape(33, 2)
                my_list.append(input_new_coords)
                self.k+=1
            else:
                break
        cap.release()

        with open(output_name, 'wb') as f:
            pickle.dump(my_list, f)

    def frame_no(self):
        frame=self.k
        return(frame)







