# gsr-analysis

### 프로젝트 설명
NIR 얼굴 영상에서 추출한 밝기값이 GSR값과 상관성이 있다는 선행연구를 바탕으로 얼굴의 영역별(이마/뺨/코) 상관성을 알아보고자 함

### 코드
1) 데이터 취득 : neulog 센서를 사용해 GSR 데이터 취득, 적외선 카메라로 얼굴 영상을 녹화함

    - 설치 : neulog api  
	- neulog_sensor.py : neulog 센서로 reference값인 GSR 데이터 취득
	- record_face.py : 얼굴 영상 녹화


2) 랜드마크 기반 얼굴 밝기값 계산 : AU_extractor - 
[Py-Feat](https://py-feat.org/content/intro.html) 라이브러리 사용

    - configs : 얼굴, 얼굴 랜드마크 검출 모델
    - onnx : 
    - face_videos : 실험을 통해 취득한 NIR 얼굴 영상
    - utils.py, face_utils.py
    - facetool.py : 얼굴 검출 후 랜드마크 기반으로 피부 영역 정의 -> 피부 밝기값, AU 계산
    - facetool_emotion.py : 얼굴 검출 후 랜드마크 기반으로 피부 영역 정의 -> 피부 밝기값, AU, emotion 계산
    - csv : facetool.py에서 계산한 피부 밝기값, AU, emotion 저장
    
3) face parsing 기반 얼굴 밝기값 계산 : [face-toolbox-keras](https://github.com/shaoanlu/face_toolbox_keras) 사용
    
    - 실행환경 : GPU, tensorflow v1 필요
    - models :  face detector, parsing 모델
    - utils : 필요한 함수들
    - videos : 실험을 통해 취득한 NIR 얼굴 영상
    - csv : 얼굴 밝기값 저장한 csv파일
    - result : face parsing 결과 이미지
    - face-brightness.py : face detect, parsing 후 피부 영역의 밝기 계산


4) 분석 