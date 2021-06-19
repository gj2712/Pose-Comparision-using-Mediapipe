import cv2
import mediapipe as mp
import time
import math
import imutils
import Module_pose as pd
import numpy as np
import pickle
from score import Score
import keypoints_from_tst as kp
s=Score()

# You can comment out below part if you already have pickle file of template stored and can directly load it

p = r'videos/templates/template_2.mp4'                     # path of the original template with which we have to compare
obj = kp.generate_vector(p)    ##### Comment out this part
output_name = 'temp_2.pkl'                                 # provide name of your pickle file
obj.form_pikle(output_name)    ##### Comment out this part
j = obj.frame_no()             ##### j will be needed so you should know the value of j to replace it in the code below
                               ##### before commenting out

# Can directly load your pickle file
with open(output_name, 'rb') as f:
    mynewlist = pickle.load(f)

cap = cv2.VideoCapture(r'videos/template_2_test/test2.mp4')     # path of the test video for which we have to find the score
cap1 = cv2.VideoCapture(p)

if (cap.isOpened() == False):
    print("Error opening test video file")
if (cap1.isOpened()== False):
    print("Error copening original template video file")

detector=pd.pose_detector()
k=0
my_list=[]
while (cap.isOpened() or cap1.isOpened()):

    success, img = cap.read()
    ret1, img1 = cap1.read()

    if success == True and ret1==True:

        img = detector.find_body(img)
        bbox,pos_list,y_list,roi_co = detector.find_position(cv2.resize(img, (372, 495)))
        pos_list.extend(y_list)
        input_points = np.array(pos_list)
        toi=detector.roi(input_points,roi_co)
        input_new_coords = np.asarray(toi).reshape(33, 2)

        my_list.append(input_new_coords)
        k+=1

        imS = cv2.resize(img, (540, 680))
        imS1 = cv2.resize(img1, (540, 680))

        cv2.imshow('Video Original', imS1)
        cv2.imshow('Video test', imS)
        cv2.waitKey(1)


    else:
        break

# When everything done, release
# the video capture object
cap.release()
my_list = np.array(my_list)

final_score,score_list = s.compare(np.asarray(my_list),np.asarray(mynewlist),j,k)
print("Similarity Score : ",int(final_score))
cv2.destroyAllWindows()
