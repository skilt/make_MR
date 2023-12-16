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

N_FFT = 1022
SAPLE_RATE = 22050
TIME_LENGTH = 8000
TARGET = "vocal_removed_mixture.wav"

for dirname, _, filenames in os.walk("D:/Py/dataset/musdb18hq"):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("\n----------------------------------------------------------------------------------------------------------------------------------")

# MusDB 데이터셋 경로 설정
musdb_path = "D:/Py/dataset/musdb18hq"

# 훈련 데이터셋 및 트랙 선택
subset = "train"  # 또는 "test"를 선택할 수 있습니다.
musdb_subset_path = os.path.join(musdb_path, subset)
tracks = os.listdir(musdb_subset_path)

i = 1
# 특정 트랙 정보 출력
for track_id in tracks:
    track_path = os.path.join(musdb_subset_path, track_id)
    print(i)
    print(f"Track ID: {track_id}")

    # Mixture파일 경로
    mixture_path = os.path.join(track_path, "mixture.wav")
    print(f"Mixture Audio Path: {mixture_path}")

    # MR 파일 경로
    mr_path = os.path.join(track_path, "vocal_removed_mixture.wav")
    print(f"MR Audio Path: {mr_path}")

    # 오디오 파일 읽어오기
    mixture_audio, sr = librosa.load(mixture_path, sr=SAPLE_RATE)
    print(mixture_audio.shape, sr)
    mr_audio, sr = librosa.load(mr_path, sr=SAPLE_RATE)
    print(mr_audio.shape, sr)

    # Mixture의 스펙트로그램 만들기
    mix_spectrogram = librosa.stft(mixture_audio, n_fft=N_FFT)
    mix_spectrogram = librosa.util.fix_length(mix_spectrogram, size=TIME_LENGTH, axis=1)
    mix_spectrogram = np.abs(mix_spectrogram)**0.5
    mix_spectrogram = np.array(mix_spectrogram).T
    mix_spectrogram = np.array(np.split(mix_spectrogram, 10, axis=0))

    # MR의 스펙트로그램 만들기
    mr_spectrogram = librosa.stft(mr_audio, n_fft=N_FFT)
    mr_spectrogram = librosa.util.fix_length(mr_spectrogram, size=TIME_LENGTH, axis=1)
    mr_spectrogram = np.abs(mr_spectrogram)**0.5
    mr_spectrogram = np.array(mr_spectrogram).T
    mr_spectrogram = np.array(np.split(mr_spectrogram, 10, axis=0))

    # Plot mix spectrogram
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(np.log1p(mix_spectrogram.reshape(-1, mix_spectrogram.shape[-1])).T, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Mix Spectrogram')
    plt.colorbar(format='%2.1f dB')

    # Plot target spectrogram
    plt.subplot(2, 1, 2)
    plt.imshow(np.log1p(mr_spectrogram.reshape(-1, mr_spectrogram.shape[-1])).T, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Target(MR) Spectrogram')
    plt.colorbar(format='%2.1f dB')

    plt.tight_layout()
    plt.show()

    i = i+1
    print("\n")
    if(i > 5):
      break