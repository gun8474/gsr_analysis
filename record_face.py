import cv2
from datetime import datetime


def record_face(save_path, FPS):
    '''
    RGB, IR카메라로 얼굴 영상 녹화
    녹화 시작 : s/S키
    녹화 종료 : ESC키
    :param save_path: 녹화한 얼굴영상 저장 경로
    :param FPS: 녹화영상 fps 설정
    :return:
    '''

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    record = False

    # 카메라를 추가적으로 연결하여 외장 카메라를 이용하는 경우 장치 번호가 1~n 까지 순차적으로 할당
    ir_capture = cv2.VideoCapture(0)
    rgb_capture = cv2.VideoCapture(1)  # cv2.CAP_DSHOW 쓰면 열리는 속도는 빨라지지만 영상이 왜곡생기고 느려짐
    # laptop_capture = cv2.VideoCapture(0)  # 카메라의 장치 번호(ID)와 연결한다. Index는 카메라의 장치 번호를 의미한다.
    print('카메라 불러옴')

    print("width: ", rgb_capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임 너비
    print("height: ", rgb_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 높이
    print("fps: ", rgb_capture.get(cv2.CAP_PROP_FPS))  # 프레임 속도

    ir_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 1920
    ir_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 1080
    rgb_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    rgb_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # laptop_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # laptop_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print('frame set완료')

    while True:
        # print('---')
        ret, frame = ir_capture.read()
        ret2, frame2 = rgb_capture.read()
        # ret3, frame3 = laptop_capture.read()

        cv2.imshow("ir_vid", frame)
        cv2.imshow('rgb_vid', frame2)
        # cv2.imshow("laptop_vid", frame3)

        # now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # 날짜, 시간 출력
        now = datetime.now().strftime("%H-%M-%S.%f")[:-3]  # 시간만 출력
        key = cv2.waitKey(33)

        # S또는 s를 누르면 시작
        if key == 83 or key == 115:
            print('녹화 시작')
            print('frame', cv2.CAP_PROP_FPS)
            record = True
            # ir_video = cv2.VideoWriter("C:/Users/Dell/Desktop/GSR/gsr/get_videos/" + str(now) + "IR.mp4", fourcc, cv2.CAP_PROP_FPS, (frame.shape[1], frame.shape[0]))
            # rgb_video = cv2.VideoWriter("C:/Users/Dell/Desktop/GSR/gsr/get_videos/" + str(now) + "RGB.mp4", fourcc,cv2.CAP_PROP_FPS, (frame2.shape[1], frame2.shape[0]))

            ir_video = cv2.VideoWriter(save_path + "IR_" + str(now) + ".mp4", fourcc, FPS,
                                       (frame.shape[1], frame.shape[0]))
            rgb_video = cv2.VideoWriter(save_path + "RGB_" + str(now) + ".mp4", fourcc, FPS,
                                        (frame2.shape[1], frame2.shape[0]))
            # laptop_video = cv2.VideoWriter(save_path + "laptop_RGB_" + str(now) + ".mp4", fourcc, FPS, (frame3.shape[1], frame3.shape[0]))

        # ESC 누르면 종료
        elif key == 27:
            print("녹화 중지")
            record = False
            break

        if record == True:
            print("녹화중")
            ir_video.write(frame)
            rgb_video.write(frame2)
            # laptop_video.write(frame3)

    ir_capture.release()
    rgb_capture.release()
    # laptop_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 얼굴 영상 녹화
    save_path = './face_videos_0601/'
    FPS = 20
    record_face(save_path, FPS)
