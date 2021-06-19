import cv2
import mediapipe as mp
import time
import math
import imutils
import numpy as np
from sklearn.preprocessing import normalize

class pose_detector():
    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode=static_image_mode
        self.model_complexity=model_complexity
        self.smooth_landmarks=smooth_landmarks
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode,
                 self.model_complexity,
                 self.smooth_landmarks,
                 self.min_detection_confidence,
                 self.min_tracking_confidence)


    def find_body(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(len(results.pose_landmarks.landmark))
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return(img)

    def find_position(self,img,draw=False):
        xlist=[]
        ylist=[]
        bbox1=[]
        bbox2=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = lm.x * w, lm.y * h
                xlist.append(cx)
                ylist.append(cy)
                # if draw:
                #     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bbox1 = int(xmin), int(ymin), int(xmax), int(ymax)
            bbox2 = [(int(xmin),int(ymin)),(int(xmax),int(ymin)),(int(xmax),int(ymax)),(int(xmin),int(ymax))]
            # cv2.rectangle(img,(bbox1[0]-20,bbox1[1]-20),(bbox1[2]+20,bbox1[3]+20),(0,255,0),2)

        return(bbox1,xlist,ylist,bbox2)

    def normalization(self,pos_list):
        norm1 = pos_list / np.linalg.norm(pos_list)
        return (norm1)


    def roi(self, imagepoints,bbox2):
        coords_new_reshaped = imagepoints
        coords_new = np.asarray(coords_new_reshaped).reshape(33, 2)
        roi_coords = bbox2
        coords_new = self.get_new_coords(coords_new, roi_coords)
        coords_new = coords_new.reshape(66, )
        return coords_new


    def get_new_coords(self, coords, fun_bound):
        coords[:, :1] = coords[:, :1] - fun_bound[0][0]
        coords[:, 1:2] = coords[:, 1:2] - fun_bound[0][1]
        return coords


def main():

    cap = cv2.VideoCapture(r'videos/templates/template_1.mp4')
    detector=pose_detector()

    while(cap.isOpened()):
        success, img = cap.read()
        if success==True:
            img = imutils.resize(img, width=520)
            img=detector.find_body(img)
            cv2.imshow("image", img)
            cv2.waitKey(1)
        else:
            break

if __name__ =='__main__':
    main()