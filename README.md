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
📦KT.PM10
 ┣ 📂Crawling
 ┃ ┣ 📂CrawlingData
 ┃ ┃ ┣ 📜sunrise_sunset_2021.csv
 ┃ ┃ ┗ 📜sunrise_sunset_test.csv
 ┃ ┣ 📂DataList
 ┃ ┃ ┣ 📜2021_datelist.csv
 ┃ ┃ ┗ 📜2022_datelist.csv
 ┃ ┗ 📜Code.ipynb
 ┣ 📂Data
 ┃ ┣ 📜air_2021.csv
 ┃ ┣ 📜air_2022.csv
 ┃ ┣ 📜test.csv
 ┃ ┣ 📜train.csv
 ┃ ┣ 📜weather_2021.csv
 ┃ ┗ 📜weather_2022.csv
 ┣ 📂Model
 ┃ ┣ 📂DataBase_For_Modeling
 ┃ ┃ ┣ 📜Baseline_results.zip
 ┃ ┃ ┣ 📜Feature_Importance.zip
 ┃ ┃ ┣ 📜Remove_not_important_features_results.zip
 ┃ ┃ ┣ 📜resid_test.zip
 ┃ ┃ ┣ 📜resid_train.zip
 ┃ ┃ ┣ 📜seasonal_test.zip
 ┃ ┃ ┣ 📜seasonal_train.zip
 ┃ ┃ ┣ 📜trend_test.zip
 ┃ ┃ ┗ 📜trend_train.zip
 ┃ ┣ 📂Model_Save
 ┃ ┃ ┣ 📜Baseline.7z
 ┃ ┃ ┗ 📜Remove_importance_variables.7z
 ┃ ┣ 📜Baseline.ipynb
 ┃ ┗ 📜Remove_not_important_variable_and_Ensemble.ipynb
 ┗ 📂Preprocessing
 ┃ ┣ 📜autocorrelation.ipynb
 ┃ ┣ 📜preprocessing.ipynb
 ┃ ┗ 📜TimeSeriesDecomposition.ipynb
```
## 파일 
- 각 폴더마다 안에 있는 readme를 통해 상세한 프로젝트 내용을 알 수 있습니다.

- Crawling : 일출, 일몰 시간을 크롤링한 과정에 대해서 다룹니다.
    - CrawlingData : 일출, 일몰시간을 크롤링한 데이터가 담겨져 있습니다.
    - DataList : 크롤링할 시간에 대한 정보를 담고 있는 데이터입니다.
    - Code.ipynb : 크롤링한 코드를 다룹니다.
    
- Data : 초기 데이터와 모델링 데이터 셋이 들어있는 폴더입니다.(초기 데이터 셋은 
    - air_2021.csv, air_2022.csv : 대기 질에 관한 정보가 들어있는 초기 데이터 셋
    - weather_2021.csv, weather_2022.csv : 날씨에 대한 정보가 들어있는 초기 데이터 셋
    - train.csv : 모델 학습을 위해 정제한 데이터 셋
    - test.csv : 예측을 위해 정제한 데이터 셋
    
- Model : 모델링에 대한 정보가 담겨있는 폴더입니다.
    - DataBase_For_Modeling : 시계열적 특성을 뽑아낸 데이터 셋
    - Model_Save : 학습시킨 모델을 저장한 파일
      - Baseline.7z : 초기 모델(variance importance를 고려하지 않고 모든 변수를 넣어서 학습시킨 모델)
      - Remove_importance_variables.7z : variance importance를 고려하여 중요한 변수만 넣어서 학습시킨 모델
    - Baseline.ipynb : 초기 모델(variance importance를 고려하지 않고 모든 변수를 넣어서 학습시킨 모델) 코딩 파일
    - Remove_not_important_variable_and_Ensemble.ipynb : variance importance를 고려하여 중요한 변수만 넣어서 학습시킨 모델의 코딩 파일
 
- Preprocessing : 데이터 전처리에 대한 정보가 담겨있는 폴더입니다.
    - autocorrelation.ipynb : 예측값(PM10)이 과거 몇 시점까지 상관성이 있는지 파악하기 위한 코드
    - preprocessing.ipynb : 데이터 전처리 함수들이 담겨있는 코드
    - TimeSeriesDecomposition.ipynb : 시계열 분해를 진행하고, 시계열 분해요소를 추출하는 과정을 담은 코드

## 문의사항
* email : ajc227ung@gmail.com
