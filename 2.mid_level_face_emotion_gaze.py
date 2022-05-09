#!/usr/bin/env python
# coding: utf-8


import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from common.utils import *
from RetinaFace.api import FaceDetection
from XiaoGaze.api import GazeEstimation
from TDDFA_V2.api import LandmarkDetection

from ArcFace.api import FaceRecognition
from TDDFA_V2.api import LandmarkDetection
from XiaoGaze.api import GazeEstimation
import json


def single_image_head_eye(face_img_path,dic):
    face_img=cv2.imread(face_img_path)
    face_embd = face_recognizer.predict(face_img)
    #print(face_embd.shape)
    face_landmarks = landmark_detector.predict(face_img)
    vis_img = landmark_detector.plot(face_img, face_landmarks)
    gaze_vector, pose_vector = gaze_estimator.predict(face_img)
    vis_img = gaze_estimator.plot(face_img, face_landmarks, gaze_vector, pose_vector)
    #plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))    
    dic[face_img_path]=[gaze_vector, pose_vector]
    return dic

def one_person(person):
    person_dic={}
    person_path=os.path.join(face_path,person)
    json_path=os.path.join(person_path,"person_gaze_pose.json")
    print(json_path)
    for image in os.listdir(person_path):
        if ".json" not in image:
            print(image)
            image_path=os.path.join(person_path,image)
            person_dic=single_image_head_eye(image_path,person_dic)
        #print(person_dic)    
    with open(json_path, 'w') as fp:
        json.dump(person_dic, fp)
    print("Done!")

if __name__ == "__main__":

    all_rename_dict = np.load('all_cilp_rename.npy', allow_pickle=True).item()
    all_rename_ls = list(all_rename_dict.values())

    original_path = '...'
    parent_path ='...'
    no_parent_path = '...'
    face_path =  '...'

    face_recognizer = FaceRecognition(device='cuda:1')
    face_detector = FaceDetection(device='cuda:1', max_face_num=5)
    gaze_estimator = GazeEstimation(device='cuda:2')
    landmark_detector = LandmarkDetection()

    for person in os.listdir(face_path)[1:]:
        one_person(person)





