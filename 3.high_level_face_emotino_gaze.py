#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np

## facial emotion：emotion_classification（7*1）, valence，arousal
def get_face(face_json):
    emotion_ls = []
    valence_ls = []
    atousal_ls = []
    for pic in sorted(list(face_json.keys())):
        emotion = face_json[pic][0]
        valence = face_json[pic][1]
        arousal = face_json[pic][2]
        emotion_ls.append(emotion)
        valence_ls.append(valence)
        atousal_ls.append(arousal)
    emotion_array = np.array(emotion_ls)
    valence_array = np.array(valence_ls)
    atousal_array = np.array(atousal_ls)
    return emotion_array,valence_array,atousal_array

## gaze: gaze_vector, pose_vector
def get_gaze(face_json):
    gaze_ls = []
    pose_ls = []
    for pic in sorted(list(face_json.keys())):
        gaze = face_json[pic][0]
        pose = face_json[pic][1]
        gaze_ls.append(gaze)
        pose_ls.append(pose)
    gaze_array = np.array(gaze_ls)
    pose_array = np.array(pose_ls)
    return gaze_array,pose_array

def get_seq_matrix_dict(prepared_dict):
    single_clip_dict = prepared_dict
    ls_dict = {} #  
    for clip_big in big_type_ls:
        ls_dict[clip_big]=[]
    for key in single_clip_dict.keys():
        big_key = key
        if big_key in big_type_one_dict.keys():
            big_value = big_type_one_dict[big_key]
            ls_dict[big_value].append(big_key)
        else:
            print(big_key)
    empty_ls = []
    for key in ls_dict.keys():
        if len(ls_dict[key])<1:
            empty_ls.append(key)
    for key in empty_ls:        
        ls_dict.pop(key)

    seq_dict = {}
    seq_matrix_dict = {}

    for clip_big in big_type_ls:
        seq_dict[clip_big]=[]
    for key in ls_dict.keys():
        for single_clip in ls_dict[key]:
            new_key = single_clip
            single_value = single_clip_dict[new_key]
            if len(single_value)>0:
                (seq_dict[key]).append(single_value)  

    for key in seq_dict.keys():
        label_ls = seq_dict[key]
        if len(label_ls)>1:
            label_matrix_value = np.vstack(label_ls)
            seq_matrix_dict[key] = label_matrix_value
        elif len(label_ls)==1:
            label_matrix_value = label_ls[0]
            seq_matrix_dict[key] = label_matrix_value
    return seq_matrix_dict

def get_mean_dict(ori_dict):
    mean_dict = {}
    for key in list(ori_dict.keys()):
        mat = ori_dict[key]
        vec = np.mean(mat, axis=0)
        mean_dict[key] = vec
    return mean_dict


if __name__ == '__main__':
    face_root_path = 'frame-face/'
    face_file = 'face_emotion.json'
    gaze_file = 'person_gaze_pose.json'

    emotion_dict = {}
    valence_dict = {}
    arousal_dict = {}

    for lecture in os.listdir(face_root_path):
        json_path = os.path.join(face_root_path,lecture,face_file)  
    #     print(json_path)
        if os.path.exists(json_path):    
            with open(json_path,'r', encoding='UTF-8') as f:
                face_json = json.load(f)
                emotion_array,valence_array,arousal_array=get_face(face_json)
                emotion_dict[lecture] = emotion_array
                valence_dict[lecture] = valence_array
                arousal_dict[lecture] = arousal_array

    gaze_dict = {}
    pose_dict = {}
    for lecture in os.listdir(face_root_path):
        json_path = os.path.join(face_root_path,lecture,gaze_file)  
    #     print(json_path)
        if os.path.exists(json_path):    
            with open(json_path,'r', encoding='UTF-8') as f:
                gaze_json = json.load(f)
                gaze_array,pose_array=get_gaze(gaze_json)
                gaze_dict[lecture] = gaze_array
                pose_dict[lecture] = pose_array
    video_id = list(gaze_dict.keys())[0]

    big_type_one_dict = np.load('big_type_one_dict_p.npy', allow_pickle=True).item()
    big_type_ls = list(set(big_type_one_dict.values()))

    emotion_final_dict = get_seq_matrix_dict(emotion_dict)
    valence_final_dict = get_seq_matrix_dict(valence_dict)
    arousal_final_dict = get_seq_matrix_dict(arousal_dict)
    gaze_final_dict = get_seq_matrix_dict(gaze_dict)
    pose_final_dict = get_seq_matrix_dict(pose_dict)
    speech_emo_mean_dict = get_mean_dict(speech_emo_dict)

    e_mean_dict = get_mean_dict(emotion_final_dict)
    v_mean_dict = get_mean_dict(valence_final_dict)
    a_mean_dict = get_mean_dict(arousal_final_dict)
    g_mean_dict = get_mean_dict(gaze_final_dict)
    p_mean_dict = get_mean_dict(pose_final_dict)

    np.save('emotion_dict.npy',e_mean_dict)
    np.save('valence_dict.npy',v_mean_dict)
    np.save('arousal_dict.npy',a_mean_dict)
    np.save('gaze_dict.npy',g_mean_dict)
    np.save('pose_dict.npy',p_mean_dict)

