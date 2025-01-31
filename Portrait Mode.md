
**사용하는 라이브러리**
- os: Operating System의 줄임말로, 운영체제에서 제공되는 여러 기능을 파이썬에서 사용할 수 있도록 함 (Ex. 디렉토리 경로 이동, 시스템 환경 변수 가져오기 등)
- urllib: URL 작업을 위한 여러 모듈을 모은 패키지. (Ex. urllib.request, urllib.parse, ...)
- cv2: OpenCV 라이브러리로, 실시간 컴퓨터 비전을 목적으로 한 프로그래밍 라이브러리
- numpy(NumPy): 행렬이나 대규모 다차원 배열을 쉽게 처리할 수 있도록 지원하는 라이브러리. 데이터 구조 외에도 수치 계산을 위해 효율적으로 구현된 기능을 제공
- pixellib: 이미지 및 비디오 segmentation을 수행하기 위한 라이브러리
- pixellib.semantic: segmentation 기법 중, semantic segmentation을 쉽게 사용할 수 있도록 만든 라이브러리
- matplotlib: 파이썬 프로그래밍 언어 및 수학적 확장 NumPy 라이브러리를 활용한 플로팅 라이브러리로, 데이터 시각화 도구

라이브러리 불러오기
```python

import os

import urllib

import cv2

import numpy as np

from pixellib.semantic import semantic_segmentation

from matplotlib import pyplot as plt

```

이미지 불러오기 
```python

# os 모듈에 있는 getenv() 함수를 이용하여 읽고싶은 파일의 경로를 file_path에 저장

# 준비한 이미지 파일의 경로를 이용하여, 이미지 파일을 읽음

# cv2.imread(경로): 경로에 해당하는 이미지 파일을 읽어서 변수에 저장

img_path = os.getenv('HOME')+'/aiffel/human_segmentation/images/my_image.png'  

img_orig = cv2.imread(img_path)

  

print(img_orig.shape)

```


이미지 색상 변환
```python

# cv2.cvtColor(입력 이미지, 색상 변환 코드): 입력 이미지의 색상 채널을 변경

# cv2.COLOR_BGR2RGB: 이미지 색상 채널을 변경 (BGR 형식을 RGB 형식으로 변경)

# plt.imshow(): 저장된 데이터를 이미지의 형식으로 표시, 입력은 RGB(A) 데이터 혹은 2D 스칼라 데이터

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html

# plt.show(): 현재 열려있는 모든 figure를 표시 (여기서 figure는 이미지, 그래프 등)

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html

plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))

plt.show()
```

모델 다운로드 : PixelLib에서 DeepLab 다운로드 받기


```python

# 저장할 파일 이름을 결정합니다

# 1. os.getenv(x)함수는 환경 변수x의 값을 포함하는 문자열 변수를 반환합니다. model_dir 에 "/aiffel/human_segmentation/models" 저장

# 2. #os.path.join(a, b)는 경로를 병합하여 새 경로 생성 model_file 에 "/aiffel/aiffel/human_segmentation/models/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5" 저장

# 1

model_dir = os.getenv('HOME')+'/aiffel/human_segmentation/models'

# 2

model_file = os.path.join(model_dir, 'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')

  

# PixelLib가 제공하는 모델의 url입니다

model_url = 'https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5'

  

# 다운로드를 시작합니다

urllib.request.urlretrieve(model_url, model_file) # urllib 패키지 내에 있는 request 모듈의 urlretrieve 함수를 이용해서 model_url에 있는 파일을 다운로드 해서 model_file 파일명으로 저장
```

세그멘테이션 모델 생성하기
```python

model = semantic_segmentation() #PixelLib 라이브러리 에서 가져온 클래스를 가져와서 semantic segmentation을 수행하는 클래스 인스턴스를 만듬


model.load_pascalvoc_model(model_file) # pascal voc에 대해 훈련된 예외 모델(model_file)을 로드하는 함수를 호출
```

모델에 이미지 입력하기
- Pascalvoc로 사전학습 된 모델 이용하기
```python

segvalues, output = model.segmentAsPascalvoc(img_path) # segmentAsPascalvoc()함 수 를 호출 하여 입력된 이미지를 분할, 분할 출력의 배열을 가져옴, 분할 은 pacalvoc 데이터로 학습된 모델을 이용
```
- 모델의 출력값 확인하기
```python

```


Pascalvoc 데이터의 라벨 종류
- background 제외 20개
- 사람 라벨의 인덱스는 15
```python
#pascalvoc 데이터의 라벨종류

LABEL_NAMES = [

    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',

    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',

    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'

]

len(LABEL_NAMES)
```