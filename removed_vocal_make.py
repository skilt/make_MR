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

# 파형을 뒤집는 것을 수행
def invert_waveform(waveform):
    inverted_waveform = -waveform
    return inverted_waveform

# train 경로: ../musdb18hq/train
# test 경로: ../musdb18hq/test
for dirname, _, filenames in os.walk("D:/Py/dataset/musdb18hq/train"):
    if (dirname == "D:/Py/dataset/musdb18hq/train"):
        continue
    else:
        mixture_path = os.path.join(dirname, "mixture.wav")
        print("해당 파일을 로드합니다: " + mixture_path)
        # 해당 오디오 파일을 로드합니다, 음원은 2채널이기 때문에 2채널로 로드합니다
        mixture, sr = librosa.load(mixture_path, sr=44100, mono=False)
        print("mixture load complete\n")

        vocal_path = os.path.join(dirname, "vocals.wav")
        print("해당 파일을 로드합니다: " + vocal_path)
        # 해당 오디오 파일을 로드합니다, 음원은 2채널이기 때문에 2채널로 로드합니다
        vocal, sr_vocal = librosa.load(vocal_path, sr=44100, mono=False)
        print("vocal load complete\n")

        # 뒤집힌 vocal을 얻습니다.
        inverted_vocal = invert_waveform(vocal)

        # 뒤집은 vocal 부분과 원래 음원을 합쳐서 vocal파트만을 상쇄시키고 vocal이 없는 instrument버전만을 뽑아냅니다
        vocal_removed_mixture = mixture + inverted_vocal

        # 결과를 저장합니다.
        sample_rate = 44100
        output_path = os.path.join(dirname, 'vocal_removed_mixture.wav')
        print("vocal_removed_mixture 버전이 해당위치에 추가됩니다: " + output_path)
        sf.write(output_path, vocal_removed_mixture.T, sample_rate, subtype='PCM_16')
        print("completed")
        print("-----------------------------------------------------------------------------------------------\n")