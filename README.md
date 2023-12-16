# 나만의 MR 만들기 프로젝트

 :point_down: 아래의 링크를 통해서도 확인해 보실 수 있습니다.  :point_down:
<https://colab.research.google.com/drive/1BVUSpwbAO8pg8J2EUVFQnBRa1XX7T-Nb?authuser=1#scrollTo=8agQ2-sB6ro6>

## 파일 설명
#### **1. model 폴더** :file_folder:
model 폴더에는 사전에 학습하여 저장된 model들이 들어 있습니다.
 - ##### model_mk1: MR을 직접 예측하여 추출하는 모델입니다.
 - ##### model_mk2: 음악의 vocal을 예측하여 추출하는 모델입니다.

#### **2. log 폴더**  :file_folder:
log 폴더에는 사전에 학습한 model들의 훈련과정이 저장된 log 파일이 들어있습니다.
 - ##### model_mk1_log: model_mk1의 log 파일입니다.
 - ##### model_mk2_log: model_mk2의 log 파일입니다.

#### **3. result 폴더**  :file_folder:
result 폴더에는 사전에 테스트한 모델별 결과가 .wav 파일로 들어있습니다.

#### **4. removed_vocal_make.py**  
Train 데이터로 사용할 MR 데이터를 생성하는 코드입니다.

#### **5. view_spectrogram.py**  
MUSDB에 있는 파일들의 스펙트로그램을 확인하는 코드입니다.

#### **6. main.py**  
모델 훈련을 수행하는 코드입니다.

#### **7. mk1_predict.py**  
MR을 직접 예측하는 model_mk1의 모델 테스트 코드입니다.

#### **8. mk2_predict.py**  
vocal을 예측하는 mode1_mk2의 모델 테스트 코드입니다.

------

## 실행 방법
 - ### **colaboratory**
     상단의 링크를 통해서 colab에서도 동작할 수 있도록 하였습니다. 저장된 MUSDB의 경로에 유의하여 순서대로 진행해 주시면 됩니다.  
     중간에 모델 학습을 진행하는 부분을 생략하여 바로 모델 테스트를 진행할 수 있습니다.  
     model 폴더안에 있는 모델들을 저장하여 colab에서 불러오면 됩니다.

    ![ddd](https://github.com/skilt/make_MR/assets/114862463/2825904e-d981-49f8-bd4b-6daad5639749)

 ------

 - ### **Pycharm**
   **1. 사전 준비**
   
     #### **실행한 환경**
        - OS: Window 11
        - Python Version: 3.10
   
     #### **설치한 라이브러리**
        import torch
        import torch.nn as nn
        import soundfile as sf
        import librosa
        import numpy as np
        import scipy.io as sio
        import matplotlib.pyplot as plt
        import os
        import torchaudio


   **2. 데이터 셋 준비**
   
     파이참 환경에서 직접 돌리실 경우에는 MUSDB 데이터 셋을 직접 다운 받으셔야 합니다.

     🔗링크: <https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav>  

     ![image](https://github.com/skilt/make_MR/assets/114862463/43c908ab-017f-495f-a77d-79f7b98bc971)

   **3. MR 데이터 준비**
   
     MUSDB 데이터 셋이 준비가 되었으면 Train 데이터로 사용할 MR 데이터를 만들어야 합니다.  
     removed_vocal_make.py를 실행하여 MR 데이터를 만듭니다.
   
     아래의 경로를 "나의 MUSDB 경로/train" 으로 지정한 뒤 실행해주고,  
   "나의 MUSDB 경로/test" 로 지정한 뒤 한번 더 실행해 주시기 바랍니다.
   
   
   ```
   # removed_vocal_make.py의 코드 일부
   for dirname, _, filenames in os.walk("D:/Py/dataset/musdb18hq/train"):
    if (dirname == "D:/Py/dataset/musdb18hq/train"):
   ```   

   **4. 실행**
   
     main.py를 실행하여 모델 학습을 진행합니다.  
     역시, 미리 학습된 모델을 불러와서 사용하는 것 또한 가능합니다.

     model_mk1을 테스트 하고자 하는 경우, mk1_predict.py를 실행하여 주시기 바랍니다.  
     model_mk2를 테스트 하고자 하는 경우, mk2_predict.py를 실행하여 주시기 바랍니다.
