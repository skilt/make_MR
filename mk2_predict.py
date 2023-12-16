# 이 코드는 보컬 추정 모델(model_mk2)에 대한 모델 예측 코드 입니다.

import torch
import torch.nn as nn
import soundfile as sf
import librosa
import numpy as np
import scipy.io as sio
import scipy.io.wavfile
import matplotlib.pyplot as plt
import soundfile as sf
import os
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import librosa.feature
from torch.utils.tensorboard import SummaryWriter
from main import UnmixModel, AudioDataset

N_FFT = 1022
SAPLE_RATE = 22050
TIME_LENGTH = 8000
TARGET = "vocal_removed_mixture.wav"

def specTowav(split_spectrogram, path):
  spectrogram = np.reshape(split_spectrogram, (-1, 512))
  spectrogram = spectrogram **2
  spectrogram = spectrogram.T
  spectrogram = librosa.griffinlim(spectrogram, n_fft=N_FFT)
  sf.write(path, spectrogram, SAPLE_RATE,  format='WAV')

# MusDB 데이터셋 경로 설정: ../musdb18hq
import random
musdb_path = "D:/Py/dataset/musdb18hq"

print("Test 샘플")
test_ds = AudioDataset("test", augmentation=False)
test_sample, _ = test_ds[49]
test_loader = DataLoader(test_ds, shuffle=False)
print(test_sample.shape)
print("---------------------------------------------------")

model = UnmixModel()
# 모델 불러오기
model.load_state_dict(torch.load("D:/Py/MR project/model/model_mk2.pth")) # 불러오는 모델에 유의!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 평가 모드
model.eval()

print("테스트 시작\n")
with torch.no_grad():
  for i, (batch_x, batch_y) in enumerate(test_loader):
    if i == 18:  # 0부터 49까지의 숫자중 아무거나 입력하여 테스트를 진행할 노래를 변경할 수 있습니다.
      # batch_x, batch_y = next(iter(test_loader))
      batch_x, batch_y = batch_x.to(device), batch_y.to(device)
      batch_x = batch_x.squeeze(0)
      batch_y = batch_y.squeeze(0)
      batch_x = batch_x.float()
      batch_y = batch_y.float()

      outputs = model(batch_x)

      input_x = batch_x.cpu().numpy()
      expect_y = batch_y.cpu().numpy()
      output = outputs.cpu().numpy()

      # Plot mix spectrogram
      plt.figure(figsize=(12, 6))
      plt.subplot(2, 1, 1)
      plt.imshow(np.log1p(expect_y.reshape(-1, expect_y.shape[-1])).T, aspect='auto', origin='lower',
                 cmap='viridis')
      plt.title('Expected Spectrogram')
      plt.colorbar(format='%+2.1f dB')

      plt.subplot(2, 1, 2)
      plt.imshow(np.log1p(output.reshape(-1, output.shape[-1])).T, aspect='auto', origin='lower', cmap='viridis')
      plt.title('Model Predicted Vocal Spectrogram') # 모델이 예측하여 뽑아낸 보컬 트랙을 보여줍니다.
      plt.colorbar(format='%+2.1f dB')

      plt.tight_layout()
      plt.show()

      # 추출한 보컬 트랙의 파형 뒤집기
      inverted_vocal = -output
      # 뒤집은 vocal 부분과 원래 음원을 합쳐서 vocal파트만을 상쇄시키고 vocal이 없는 instrument버전만을 뽑아냅니다
      vocal_removed_mixture = input_x + inverted_vocal

      # Pycharm 코드에서는 예측 결과를 저장합니다. 저장 위치에 유의하시기 바랍니다.
      specTowav(input_x, "D:/Py/MR project/result/mk2_input.wav")
      #specTowav(expect_y, "D:/Py/MR project/result/mk2_expected.wav") # 기대 출력 결과 입니다.
      specTowav(vocal_removed_mixture, "D:/Py/MR project/result/mk2_predict.wav") # 실제 모델이 예측하여 생성한 MR 파일입니다.
      break