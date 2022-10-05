# 코미디 빅리그 코너 구분 모델

## 개요
* 본 프로젝트는 KT 에이블스쿨 3차 미니프로젝트에서 진행하였습니다.

## 배경
* 올레 TV 예능 콘텐츠 중 하나인 프로그램 "코미디 빅리그"의 영상을 분석하고 코너를 분류하는 AI 모델을 만드는 것이 우리의 목표였습니다.
* "코미디 빅리그"는 tvN에서 방영 중인 서바이벌 형식의 공개 코미디 프로그램으로 3개월 단위로 인기 코너를 선정하는 방식인 쿼터제를 도입하고 있습니다.
* 우리는 2022년 1쿼터(1~3월)방영분 중 2월 방영한 2편을 학습데이터로 하여 3월 방송의 코너 구간을 추출해내었습니다.

## 개발환경
<p align="center">
  <img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=TensorFlow&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/scikit-learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/-matplotlib-blue"/></a>&nbsp
</p>

## 프로젝트 결과
- Accuracy  : 100%
- 전체 분류 결과의 경우, 저작권 침해의 여부가 있기 때문에, 배너를 전처리하고 예측한 영상만을 게시하는 점 양해부탁드립니다.
- 실제로는 각 코너가 나오지 않는 부분까지 모두 예측했고, 그 결과 Accuracy 100%를 달성했습니다.

https://user-images.githubusercontent.com/89781598/193985854-3bec5847-c038-48d6-a8ed-8cb5f121437e.mp4

## 데이터
![image](https://user-images.githubusercontent.com/89781598/193987096-c9b50744-2285-4b3f-ad20-0ddde98dc36b.png)

* Train : 2월 방영분 2편
* Test  : 3월 방영분 1편

## 데이터 전처리
![image](https://user-images.githubusercontent.com/89781598/193987209-1f3edea9-8693-4904-a084-ef622ad91cd6.png)

- 해당 프로젝트는 최대한 많은 이미지 전처리 기법을 다룰려고 하였습니다.
- 사용했던 전처리 방법은 OCR Contour, LapSRNx8, Center Crop입니다.

## Contour LapSRN to CRNN + Yolov5 detected vector Model
![image](https://user-images.githubusercontent.com/89781598/193987343-e6126f34-7153-4493-9de3-4cbdda67b3d7.png)

## 모델링 아이디어
![image](https://user-images.githubusercontent.com/89781598/193987250-1259d195-909a-4679-969b-39665732bf83.png)

## 파일 구조
```
📦KT.Comic
 ┣ 📂code
 ┃ ┣ 📜Extraction.py
 ┃ ┣ 📜Model.py
 ┃ ┣ 📜MoveFile.py
 ┃ ┣ 📜predict.py
 ┃ ┣ 📜train.py
 ┃ ┣ 📜Video2Image.py
 ┃ ┗ 📜YoloMatrix.py
```
## 파일 
- Extraction.py : 이미지 전처리 함수들이 들어있습니다.
- Model.py : 모델링에 대한 함수들이 들어있습니다.
- MoveFile.py : 파일 이동에 대한 함수들이 들어있습니다.
- predict.py : 비디오를 코너별로 구분하는 코드입니다.
- train.py : 모델을 학습하는 코드입니다.
- Video2Image.py : 비디오를 이미지로 바꿔주는 파일입니다.
- YoloMatrix.py : Yolov5를 이용하여 추출된 객체들의 갯수를 벡터화시키는 함수들이 들어있습니다.

## 참고사항
- train.py와 predict.py는 Video2Image.py에서 비디오에서 추출된 이미지가 들어있는 폴더의 경로를 넣어준 후, 파라미터를 맞춰 코드를 돌리면 자동 학습, 예측이 진행됩니다.

## 문의사항
* email : ajc227ung@gmail.com
