import os
import cv2
import numpy as np

parent_array = np.load('array_parent_video_ls.npy')
parent_ls = list(parent_array)

def FrameSampleVideo(video_path,frame_folder):
    cap = cv2.VideoCapture(video_path)
    isOpened = cap.isOpened 

    n_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),500)
#         n_frame = 500                           
    fps = cap.get(cv2.CAP_PROP_FPS) 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # w
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # h

    i = 0 
    frameFrequency = int(n_frame/10)
    while(isOpened):
        if i >=  n_frame:
            break
        else:
            i = i+1
        (flag,frame) = cap.read() 
        fileName = 'image'+str(i)+'.jpg' 
        outPutDirName = frame_folder  
        if not os.path.exists(outPutDirName):
            os.makedirs(outPutDirName)
        if i % frameFrequency == 0:
#             print(fileName)
            cv2.imwrite(outPutDirName+'/'+fileName,frame,[cv2.IMWRITE_JPEG_QUALITY,100])
    print('FINISH', frame_folder)


def main():
    original_path = '...'
    output_path = '...'
    for parent_video in parent_ls:
        parent_video_path = os.path.join(original_path,parent_video.split('_')[0],parent_video+'.mp4')
        if not os.path.exists(parent_video_path):
            print(parent_video_path)
        output_video_frame_path = os.path.join(output_path,parent_video)
        if not os.path.exists(output_video_frame_path):
            os.mkdir(output_video_frame_path)
        FrameSampleVideo(parent_video_path,output_video_frame_path)

from time import strftime, localtime

if __name__ == "__main__":
    print('========Main starts======',strftime("%Y-%m-%d %H:%M:%S", localtime()))
    main()
    print('========Main ends========',strftime("%Y-%m-%d %H:%M:%S", localtime()))
