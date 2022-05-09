#!/usr/bin/env python
# coding: utf-8

import os
from moviepy.editor import *

def VideoToAudio(inputpath,outputpath):
    video = VideoFileClip(inputpath)
    audio = video.audio
    audio.write_audiofile(outputpath)

def main(rootDir,outputfolder):
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            print(root)
            out_root = root.replace('teamwork','teamwork-speech')
            if  not os.path.exists(out_root):
                os.makedirs(out_root)
            
            file_name = os.path.join(root,file)
            print(file_name)
            outputpath = file_name.replace('teamwork','teamwork-speech')
            outputpath = outputpath.replace('mp4','wav')
            print(outputpath)
                    
            if not os.path.exists(outputpath):
                VideoToAudio(file_name,outputpath)

if __name__ == '__main__':
    outputfolder = '...'
    rootDir= '...'
    main(rootDir,outputfolder)
