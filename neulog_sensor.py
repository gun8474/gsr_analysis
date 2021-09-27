import requests
import pandas as pd
from datetime import datetime

def neulog_sensor(save_path, file_name):
    '''
    neulog api로 GSR값 측정
    코드 실행 방법 : 센서 usb를 노트북에 꽂고, neulog api를 실행시킨 후, sensor.py 코드 실행
                   실행할 때마다 센서와 neulog api를 껐다 켜야함
    출력 :GSR값, 현재시간이 담긴 csv파일
    :param save_path: neulog 데이터 저장 경로
    :param file_name: 파일명
    :return:
    '''


    # 버전 확인
    url_version = "http://localhost:22002/NeuLogAPI?GetServerVersion"
    response_version = requests.get(url_version)
    print("\nserver version \nstatus code : ", response_version.status_code)
    print("answer : ", response_version.text)

    # 서버 상태
    url_status = "http://localhost:22002/NeuLogAPI?GetSeverStatus"
    response_status = requests.get(url_status)
    print("\nserver status \nstatus code : ", response_status.status_code)
    print("answer : ", response_status.text)

    # 센서 아이디 1로 설정
    url_id = "http://localhost:22002/NeuLogAPI?SetSensorsID:[1]"
    # params = {'param': '1'}
    response_id = requests.get(url_id)
    print("\nsensor ID \nstatus code : ", response_id.status_code)
    print("answer : ", response_id.text)

    # Start Experiment
    # 4-1000fps, 5-100fps, 6-50fps, 7-20fps, 8-10fps, 9-5fps, 10-2fps, 11-1fps
    # 3번째 파라미터-fps
    # 4번째 파라미터-측정할 gsr데이터의 개수, 20 fps로 대략 10분동안 측정하면 12000개의 데이터를 측정함, 넉넉히 5만으로 설정함
    url_start = "http://localhost:22002/NeuLogAPI?StartExperiment:[GSR],[1],[7],[50000]"
    # params_start = {'param1': 'GSR', 'param2': '7'}
    response_start = requests.get(url_start)
    print("\nstart experiment \nstatus code : ", response_start.status_code)
    print("answer : ", response_start.text)

    # 데이터 읽기_중요_실질적으로 데이터 얻는 곳
    url_get = "http://localhost:22002/NeuLogAPI?GetExperimentSamples"
    time_list = []

    while (requests.get(url_get).json()):
        # print("데이터 취득중")
        value = requests.get(url_get).json()
        text = value['GetExperimentSamples']
        now = datetime.now().strftime("%H-%M-%S.%f")[:-3]

        # 20 fps로 설정하면 타임스탬프 1개가 찍힐때(4초) 대략 84개의 gsr값을 가져옴
        for i in range(84):
            time_list.append(now)

        length = len(text[0][1:])
        textframe = pd.DataFrame(text[0][1:])
        timeframe = pd.DataFrame(time_list)

        # print("value, time: ", length, text[0][1:], now)
        print("길이: ", length)
        # textframe.to_csv("./neulog_data/sw_value.csv", header=False, index=False)  # csv파일로 저장
        # timeframe.to_csv("./neulog_data/sw_time.csv", header=False, index=False)  # csv파일로 저장
        textframe.to_csv(neulog_path + file_name + "_value.csv", header=False, index=False)  # csv파일로 저장
        timeframe.to_csv(neulog_path + file_name + "_time.csv", header=False, index=False)  # csv파일로 저장


    # Stop Experiment
    url_stop = "http://localhost:22002/NeuLogAPI?StopExperiment"
    response_stop = requests.get(url_stop)
    print("\nstop experiment \nstatus code : ", response_stop.status_code)
    print("answer : ", response_stop.text)


if __name__ == "__main__":
    # neulog 데이터 취득
    neulog_path = './neulog_data/'
    file_name = 'sg_0911'
    neulog_sensor(neulog_path, file_name)

