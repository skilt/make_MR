# 나만의 MR 만들기 프로젝트

 :point_down: 아래의 링크를 통해서도 확인해 보실 수 있습니다.  :point_down:
<https://colab.research.google.com/drive/1BVUSpwbAO8pg8J2EUVFQnBRa1XX7T-Nb?authuser=1#scrollTo=8agQ2-sB6ro6>

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
   
     파이썬 환경에서 직접 돌리실 경우에는 MUSDB 데이터 셋을 직접 다운 받으셔야 합니다.

     링크: <https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav>  

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
