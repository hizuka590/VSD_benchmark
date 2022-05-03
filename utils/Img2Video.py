import os
import cv2
from PIL import Image
import numpy as np

img_dir = '/opt/sdb/polyu/VSD_dataset/train/images'  #to be modified
video_dir = '/opt/sdb/polyu/VSD_dataset/train_video/RGB'
fps = 30

file_url=[]
dir_url=[]

def write_video(file_name, images, slide_time=0 ,h=720,w=1280):
    # fourcc = cv2.VideoWriter_fourcc(*'XVID') #for .mov format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_name, fourcc, fps, (h,w))

    for image in images:
        cv_img = cv2.imread(image)
        # cv_img = cv2.resize(cv_img, (w,h))
        out.write(cv_img)
    out.release()

for root , dirs, files in os.walk(img_dir):

    for dir in dirs:

        curr_dir = os.path.join(root, dir)
        dir_url.append(curr_dir)
        dir_prefix = dir
        # print('current dir:', curr_dir)
        video_save = ''
        h,w = 0,0
        for _,_,files in os.walk(curr_dir):
            # print('files:',(files)) # not sorted
            imgs = []
            for img in sorted(files):
                imgs.append(os.path.join(curr_dir, img))
            # print('img lists:',imgs)
        w,h,_ = cv2.imread(imgs[0]).shape
        video_save = video_dir + "/" + dir_prefix + ".mp4"
        print('video dir:{} with shape w {},h {}'.format(video_save,w,h))
        write_video(video_save,imgs,0,h=h,w=w)

# raw = cv2.VideoCapture('/opt/sdb/polyu/VSD_dataset/train_video/RGB/baby_wave1.mov')
#
# while 1:
#     ret, frame = raw.read()
#     cv2.imshow('hsv', frame)
#     cv2.waitKey(10)
