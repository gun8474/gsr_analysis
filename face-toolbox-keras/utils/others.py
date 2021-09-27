import os
import cv2
import numpy as np


# 이미지 사이즈 조정
def resize_image(im, max_size=320):
    if np.max(im.shape) > max_size:
        ratio = max_size / np.max(im.shape)
        # print(f"Resize image to ({str(int(im.shape[1]*ratio))}, {str(int(im.shape[0]*ratio))}).")
        return cv2.resize(im, (0, 0), fx=ratio, fy=ratio)
    return im


# 폴더 생성
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)