'''
key frame extraction 
author: Jayden Luan
mechanism: frame diffs
'''

import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema
import seaborn as sns
from random import sample
import os 

def extract_from_video(num = 15, video_path = None, output_path = None):
    '''
    num: number of frames to extract from a video
    video_path: path of video
    '''
    if not video_path:
        print('video_path is not assigned')
        return
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        print(video_path, 'is successfully opened')
    else:
        return

    curr_frame = None
    prev_frame = None

    frame_diffs = []
    frames = []
    ret, frame = cap.read()
    i = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # luv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luv = frame[150:, :]
        # luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)

        # luv = frame
        curr_frame = luv
        if i == 1:
            print(curr_frame.shape)
        if curr_frame is not None and prev_frame is not None:
            #logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            if i == 2:
                print(diff.shape)
            count = np.sum(diff)
            frame_diffs.append(count)
            # frame = Frame(i, frame, count)
            frames.append(frame)
        prev_frame = curr_frame
        i = i + 1
        # print(i)

        cv2.imshow('frame',luv)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # def smooth(x, window_len=13, window='hanning'):
    
    #     print(len(x), window_len)

    #     s = np.r_[2 * x[0] - x[window_len:1:-1],
    #               x, 2 * x[-1] - x[-1:-window_len:-1]]

    #     if window == 'flat':  
    #         w = np.ones(window_len, 'd')
    #     else:
    #         w = getattr(np, window)(window_len)
    #     y = np.convolve(w / w.sum(), s, mode='same')
    #     return y[window_len - 1:-window_len + 1]

    # a_smooth = smooth(np.array(frame_diffs), window_len=13)
    # print(i)
    # print(len(frame_diffs))
    # print(len(a_smooth))

    threshold_upper = np.percentile(frame_diffs, 98)
    threshold_lower = np.percentile(frame_diffs, 85)
    extract_ind = []
    # extract_diff = []
    for i in range(len(frame_diffs)):
        if frame_diffs[i] > threshold_upper or frame_diffs[i] < threshold_lower:
            pass
        else:
            extract_ind.append(i)
            # extract_diff.append(frame_diffs[i])
    # print(len(extract_diff))
    # print('extract ratio:', 98 - 85, '%')
    # plt.plot(extract_diff)
    # plt.plot(frame_diffs)
    # plt.hlines(np.percentile(frame_diffs, 97), 0, 1400)
    # plt.hlines(np.percentile(frame_diffs, 65), 0 , 1400)
    # sns.kdeplot(frame_diffs)
    # plt.vlines(1190000, 0, 0.000001)
    # plt.vlines(520000, 0, 0.000001)
    # plt.show()


    ## save
    dir_name = video_path.split('\\')[-1]
    sampled_extract_ind = sample(extract_ind, num)
    if not os.path.exists(os.path.join(output_path, dir_name)):
        os.mkdir(os.path.join(output_path, dir_name))
    for i in sampled_extract_ind:
        name = "frame_" + str(i) + ".jpg" 
        cv2.imwrite(os.path.join(output_path, dir_name, name), frames[i])

# videos = os.listdir('test')
# for i in videos:
#     print(os.path.join('test', i))
def extract_from_dir(input_dir_name = 'test', output_dir_name = 'output_test'):
    cur_path = os.getcwd()
    input_path = os.path.join(cur_path, input_dir_name)
    output_path = os.path.join(cur_path, output_dir_name)
    videos = os.listdir(input_path)
    for video_name in videos:
        extract_from_video(15, os.path.join(input_path, video_name), output_path)

if __name__ == '__main__':
    extract_from_dir(input_dir_name='test', output_dir_name='output_test')