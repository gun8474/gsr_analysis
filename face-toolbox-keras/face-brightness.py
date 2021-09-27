import os
import cv2
import csv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from models.detector import face_detector
from models.parser import face_parser
from utils.visualize import show_parsing_with_annos
from utils.others import resize_image, createFolder

print(tf.__version__)


# face parsing 후 평균 얼굴 밝기 계산
def bright(im):
    # 1) resize
    img = resize_image(im)  # Resize image to prevent GPU OOM.
    h, w, _ = img.shape

    # 2) face detection
    bboxes = fd.detect_face(img, with_landmarks=False)
    assert len(bboxes) > 0, "No face detected."

    # Display detected face
    x0, y0, x1, y1, score = bboxes[0]  # show the first detected face
    x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
    face = img[x0:x1, y0:y1, :]

    # 3) face parsing
    out = prs.parse_face(img)
    out_face = prs.parse_face(face)

    # 4) 피부영역 추출
    mask1 = [out_face[0] == 1]  # skin
    mask2 = [out_face[0] == 10]  # nose
    mask = np.logical_or(mask1, mask2)
    mask = np.array(mask, dtype=np.uint8)  # 피부면 1, 아니면 0이 들어있음
    skin = cv2.bitwise_and(face, face, mask=mask[0])

    # 5) 피부 평균 밝기 계산
    skin_num = mask.sum()
    skin_sum = skin.sum() / 3
    mean = skin_sum / skin_num
    # print("피부 밝기 합: ", skin_sum)
    # print("피부 픽셀의 개수: ", skin_num)
    print("얼굴 평균 밝기: {:.3f}".format(mean))

    return out, out_face, skin, mask, mean


if __name__ == "__main__":
    # load models
    fd = face_detector.FaceAlignmentDetector(lmd_weights_path="./models/detector/FAN/2DFAN-1_keras.h5")  # 1이 가장 속도가 빠름
    prs = face_parser.FaceParser()

    # 영상 1개만
    video_path = 'videos/금비_IR.mp4'
    file_name = video_path[7:9]
    cap = cv2.VideoCapture(video_path)

    curr_path = os.getcwd()
    file_path = curr_path + '/result/' + file_name
    createFolder(file_path)
    print("file path: ", file_path)

    frame_num = 0
    while cap.isOpened():
        success, orig_image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        print("frame num: ", frame_num)
        out, out_face, skin, mask, mean = bright(orig_image)

        plt.imsave(file_path + '/' + str(frame_num) + '_face.jpg', out_face[0])
        plt.imsave(file_path + '/' + str(frame_num) + '_skin.jpg', skin)  # 피부영역 저장

        # 6) 얼굴 밝기 저장
        f = open('csv/' + file_name + '.csv', 'a', newline='')
        wr = csv.writer(f)
        wr.writerow([mean])
        f.close()

        frame_num += 1
        cv2.destroyAllWindows()  # 리소스 해제

    """
    # 영상 여러개 - 제대로 되는지 잘 모르겠음
    video_folder = 'videos/'
    video_list = os.listdir(video_folder)
    print("video list: ", video_list)
    for video in video_list:

        file_name = video[0:2]
        curr_path = os.getcwd()
        file_path = curr_path + '/result/' + file_name
        createFolder(file_path)

        video_path = curr_path + '/' + video_folder + video
        cap = cv2.VideoCapture(video_path)

        print("file path: ", file_path)
        print("video path: ", video_path)

        frame_num = 0
        while cap.isOpened():
            success, orig_image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            print("frame num: ", frame_num)
            out, out_face, skin, mask, mean = bright(orig_image)

            # plt.imsave(file_path + '/' + str(frame_num) + '_all.jpg', out[0])  # 결과 이미지 저장
            plt.imsave(file_path + '/' + str(frame_num) + '_face.jpg', out_face[0])
            plt.imsave(file_path + '/' + str(frame_num) + '_skin.jpg', skin)  # 피부영역 저장

            # 6) 얼굴 밝기 저장
            f = open('csv/' + file_name + '.csv', 'a', newline='')
            wr = csv.writer(f)
            wr.writerow([mean])
            f.close()


            frame_num += 1
            cv2.destroyAllWindows()  # 리소스 해제
    """
