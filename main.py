# 실행 전, 특정 모델만 훈련 하려는 경우 주석 처리를 해 주시기 바랍니다.
# 1. MR 추정 모델(model_mk1)을 훈련한 뒤 모델을 저장하려는 경우는 코드의 379 ~ 503 라인을 전체 주석 처리하여 주시기 바랍니다.
# 2. 보컬 추정 모델(model_mk2)을 훈련한 뒤 모델을 저장하려는 경우는 코드의 275 ~ 376 라인을 전체 주석 처리하여 주시기 바랍니다.

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
TARGET = "vocal_removed_mixture.wav" # 나중에 보컬추출모델에 적용할 때는 이 부분을 vocals.wav로 변환

# Sound Augmentation
# 출처: https://www.kaggle.com/code/huseinzol05/sound-augmentation-librosa/notebook

def change_pitch_and_speed(samples, random_uniform):
    y_pitch_speed = samples.copy()
    # length_change = np.random.uniform(low=0.8, high = 1)
    length_change = random_uniform
    speed_fac = 1.0  / length_change
    tmp = np.interp(np.arange(0,len(y_pitch_speed),speed_fac),np.arange(0,len(y_pitch_speed)),y_pitch_speed)
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed *= 0
    y_pitch_speed[0:minlen] = tmp[0:minlen]
    return y_pitch_speed

def value_augmentation(samples, random_uniform):
    y_aug = samples.copy()
    # dyn_change = np.random.uniform(low=1.5,high=3)
    dyn_change = random_uniform
    y_aug = y_aug * dyn_change
    return y_aug

def add_distribution_noise(samples, random_uniform):
    y_noise = samples.copy()
    noise_amp = 0.005*random_uniform*np.amax(y_noise)
    y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])
    return y_noise

def streching(samples):
    input_length = len(samples)
    streching = samples.copy()
    streching = librosa.effects.time_stretch(streching.astype('float'), rate=1.1)
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
    return streching

def change_pitch_only(samples, random_uniform):
    y_pitch = samples.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*random_uniform
    y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'),
                                      sr=SAPLE_RATE, n_steps=pitch_change,
                                      bins_per_octave=bins_per_octave)
    return y_pitch
def change_speed_only(samples, random_uniform):
    y_speed = samples.copy()
    # speed_change = np.random.uniform(low=0.9,high=1.1)
    speed_change = random_uniform
    tmp = librosa.effects.time_stretch(y_speed.astype('float64'), rate = speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed *= 0
    y_speed[0:minlen] = tmp[0:minlen]
    return y_speed

def random_shifting(samples, random_uniform):
    y_shift = samples.copy()
    timeshift_fac = 0.2 *2*(random_uniform-0.5)  # up to 20% of length
    start = int(y_shift.shape[0] * timeshift_fac)
    if (start > 0):
        y_shift = np.pad(y_shift,(start,0),mode='constant')[0:y_shift.shape[0]]
    else:
        y_shift = np.pad(y_shift,(0,-start),mode='constant')[0:y_shift.shape[0]]
    return y_shift

def hpss_harmonics(samples):
    y_harm, y_perc = librosa.effects.hpss(samples.astype('float64'))
    return y_harm

def hpss_percussive(samples):
    y_harm, y_perc = librosa.effects.hpss(samples.astype('float64'))
    return y_perc

# aug_num값에 따른 Augmentation 수행
def audio_aug(mix_audio, target_audio, aug_num):
    index = aug_num % 10
    mix_aug_audio = mix_audio
    target_aug_audio = target_audio

    if index == 0:
        mix_aug_audio = mix_audio
        target_aug_audio = target_audio

    elif index == 1:
        ran = np.random.uniform(low=0.8, high = 1)
        mix_aug_audio = change_pitch_and_speed(mix_audio, ran)
        target_aug_audio = change_pitch_and_speed(target_audio, ran)

    elif index == 2:
        ran = np.random.uniform(low=1.5,high=3)
        mix_aug_audio = value_augmentation(mix_audio, ran)
        target_aug_audio = value_augmentation(target_audio, ran)

    elif index == 3:
        ran = np.random.uniform()
        mix_aug_audio = add_distribution_noise(mix_audio, ran)
        target_aug_audio = add_distribution_noise(target_audio, ran)

    elif index == 4:
        mix_aug_audio = streching(mix_audio)
        target_aug_audio = streching(target_audio)

    elif index == 5:
        ran = np.random.uniform()
        mix_aug_audio = change_pitch_only(mix_audio, ran)
        target_aug_audio = change_pitch_only(target_audio, ran)

    elif index == 6:
        ran = np.random.uniform(low=0.9,high=1.1)
        mix_aug_audio = change_speed_only(mix_audio, ran)
        target_aug_audio = change_speed_only(target_audio, ran)

    elif index == 7:
        ran = np.random.uniform()
        mix_aug_audio = random_shifting(mix_audio, ran)
        target_aug_audio = random_shifting(target_audio, ran)

    elif index == 8:
        mix_aug_audio = hpss_harmonics(mix_audio)
        target_aug_audio = hpss_harmonics(target_audio)

    elif index == 9:
        mix_aug_audio = hpss_percussive(mix_audio)
        target_aug_audio = hpss_percussive(target_audio)

    return np.array(mix_aug_audio), np.array(target_aug_audio)

# Audio를 가져오는 함수
def getAudio(file_path, type="mixture.wav"):
    mixture_path = os.path.join(file_path, type)
    print(mixture_path)
    if os.path.exists(mixture_path):
        audio, sr = librosa.load(mixture_path, sr=SAPLE_RATE)
        return audio
    else:
        print(f"Warning: File not found - {mixture_path}")
        return None


def getSpectrogram(audio):
    # defualt_librosa_sample_rate: 22050
    # defualt_librosa_nfft: 512
    # stft_time_length: 12000 = 6143488 / 512(nfft) + 1
    # stft_time_length: 6000 = 3071488 / 512(nfft) + 1
    # stft_time_length: 8000 = 4095488 / 512(nfft) + 1
    # 6143488 / 22050 = 4.6 분
    # 3071488 / 22050 = 2.3 분
    spectrogram = librosa.stft(audio, n_fft=N_FFT)
    spectrogram = librosa.util.fix_length(spectrogram, size=TIME_LENGTH, axis=1)
    spectrogram = np.abs(spectrogram) ** 0.5
    spectrogram = np.array(spectrogram).T
    spectrogram = np.array(np.split(spectrogram, 10, axis=0))
    return spectrogram

def augAudio(file_path, aug_num):
    mix_audio = getAudio(file_path, type="mixture.wav")
    target_audio = getAudio(file_path, type=TARGET)

    mix_aug_audio, target_aug_audio = audio_aug(mix_audio, target_audio, aug_num)
    mix_spectrogram = getSpectrogram(mix_aug_audio)
    target_sepctrogram = getSpectrogram(target_aug_audio)
    return mix_spectrogram, target_sepctrogram

# MUSDB 데이터셋 경로 설정: ../musdb18hq
import random
musdb_path = "D:/Py/dataset/musdb18hq"

# 스펙트로그램 데이터셋 만들기
class AudioDataset(Dataset):
    def __init__(self, subset=["train","test"],augmentation=True):
        self.musdb_path = os.path.join(musdb_path, subset)
        self.tracks = os.listdir(self.musdb_path)
        self.augmentation = augmentation

    def __getitem__(self, index):
        track_id = self.tracks[index]
        track_path = os.path.join(self.musdb_path, track_id)
        original_audio_path = os.path.join(track_path, "mixture.wav")
        vocal_removed_audio_path = os.path.join(track_path, TARGET)
        aug_num = 0
        if self.augmentation:
          aug_num = random.randint(0, 4)
        mix_spectrogram, target_sepctrogram = augAudio(track_path, aug_num)
        return np.array(mix_spectrogram), np.array(target_sepctrogram)

    def __len__(self):
        return len(self.tracks)

train_ds = AudioDataset("train")
test_ds = AudioDataset("test", augmentation=False)

total_samples = len(train_ds)
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_ds, [train_size, val_size])

# 샘플 가져오기
print("Train 샘플")
train_sample,_ = train_ds[99]
print(train_sample.shape)
print("---------------------------------------------------")
# test_sample,_ = test_ds[49]
# print(test_sample.shape)

# 모델 설계
class UnmixModel(nn.Module):
    def __init__(self):
        super(UnmixModel, self).__init__()

        self.layer_norm = nn.LayerNorm(512)
        self.dense1 = nn.Linear(512, 512)
        self.batch_norm1 = nn.BatchNorm1d(800)
        self.activation1 = nn.Tanh()
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.dense2 = nn.Linear(512, 512)
        self.batch_norm2 = nn.BatchNorm1d(800)
        self.activation2 = nn.ReLU()
        self.dense3 = nn.Linear(512, 512)
        self.batch_norm3 = nn.BatchNorm1d(800)
        self.activation3 = nn.ReLU()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.dense1(x)
        x = self.batch_norm1(x)
        x_skip = self.activation1(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x + x_skip
        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)
        x = self.dense3(x)
        x = self.batch_norm3(x)
        x = self.activation3(x)

        return x * x

print("GPU 확인")
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
print("---------------------------------------------------\n")

# model_mk1 모델 훈련(MR 추측 모델)
if __name__ == "__main__":
    model = UnmixModel()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Model 요약
    print("사용할 모델")
    print(model)
    print("\n\n")

    # Create DataLoader instances for training and test sets
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=False)
    test_loader = DataLoader(test_ds, shuffle=False)

    # GPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    num_epochs = 80 #60
    # 로그 확인용
    writer = SummaryWriter("model_mk1_log")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        i = epoch + 1
        print(i,"번째 진행중", "--------------------------------------Train 시작-------------------------------------------")

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Print the shape of batch_x
            batch_x = batch_x.squeeze(0)
            batch_y = batch_y.squeeze(0)
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            # print("Batch X Shape:", batch_x.shape)
            # print("Batch Y Shape:", batch_y.shape)

            optimizer.zero_grad()
            outputs = model(batch_x)

            # Move outputs to CPU before using numpy
            predict_y = batch_y.cpu().detach().numpy()
            output = outputs.cpu().detach().numpy()

            # print("output Shape:", output.shape)
            print("------------------------------------------------------------")

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            average_loss = total_loss / len(train_loader)

            # if(count == 10):
            #     # Plot mix spectrogram
            #     plt.figure(figsize=(12, 6))
            #     plt.subplot(2, 1, 1)
            #     plt.imshow(np.log1p(predict_y.reshape(-1, predict_y.shape[-1])).T, aspect='auto', origin='lower',
            #                cmap='viridis')
            #     plt.title('Predict Spectrogram')
            #     plt.colorbar(format='%+2.0f dB')
            #
            #     plt.subplot(2, 1, 2)
            #     plt.imshow(np.log1p(output.reshape(-1, output.shape[-1])).T, aspect='auto', origin='lower',
            #                cmap='viridis')
            #     plt.title('Output Spectrogram')
            #     plt.colorbar(format='%+2.0f dB')
            #
            #     plt.tight_layout()
            #     plt.show()
        writer.add_scalar('Train Loss', average_loss, epoch)
        print("\n---------------------------------------Train 완료, Validation 시작-------------------------------------------")

        #검증 loop
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for val_batch_x, val_batch_y in val_loader:
                val_batch_x, val_batch_y = val_batch_x.to(device), val_batch_y.to(device)
                val_batch_x = val_batch_x.squeeze(0).float()
                val_batch_y = val_batch_y.squeeze(0).float()

                val_outputs = model(val_batch_x)
                print("------------------------------------------------------------")
                val_loss = criterion(val_outputs, val_batch_y)
                total_val_loss += val_loss.item()

                average_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar('Validation Loss', average_val_loss, epoch)
        print("\n")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_loss:.4f}, Validation Loss: {average_val_loss:.4f}")
        print("\n")

    writer.close()
    torch.save(model.state_dict(), "D:/Py/MR project/model/model_mk1.pth") #모델 저장 위치에 유의!
    print("model_mk1 생성 완료\n")
    print("-----------------------------------------------------------------------------------------")

    # model_mk2 모델 훈련(보컬 추측 모델)
    print("여기서 부터 model_mk2 모델 훈련을 시작합니다.\n")
    TARGET = "vocals.wav"  # 나중에 보컬추출모델에 적용할 때는 이 부분을 vocals.wav로 변환

    train_ds = AudioDataset("train")
    test_ds = AudioDataset("test", augmentation=False)

    total_samples = len(train_ds)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_ds, [train_size, val_size])

    # 샘플 가져오기
    print("Train 샘플")
    train_sample, _ = train_ds[99]
    print(train_sample.shape)
    print("---------------------------------------------------")
    # test_sample,_ = test_ds[49]
    # print(test_sample.shape)

    print("GPU 확인")
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    print("---------------------------------------------------\n")

    model = UnmixModel()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Model 요약
    print("사용할 모델")
    print(model)
    print("\n\n")

    # Create DataLoader instances for training and test sets
    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=False)
    test_loader = DataLoader(test_ds, shuffle=False)

    # GPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    num_epochs = 80 #60
    # 로그 확인용
    writer = SummaryWriter("model_mk2_log")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        i = epoch + 1
        print(i,"번째 진행중", "--------------------------------------Train 시작-------------------------------------------")

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Print the shape of batch_x
            batch_x = batch_x.squeeze(0)
            batch_y = batch_y.squeeze(0)
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            # print("Batch X Shape:", batch_x.shape)
            # print("Batch Y Shape:", batch_y.shape)

            optimizer.zero_grad()
            outputs = model(batch_x)

            # Move outputs to CPU before using numpy
            predict_y = batch_y.cpu().detach().numpy()
            output = outputs.cpu().detach().numpy()

            # print("output Shape:", output.shape)
            print("------------------------------------------------------------")

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            average_loss = total_loss / len(train_loader)

            # if(count == 10):
            #     # Plot mix spectrogram
            #     plt.figure(figsize=(12, 6))
            #     plt.subplot(2, 1, 1)
            #     plt.imshow(np.log1p(predict_y.reshape(-1, predict_y.shape[-1])).T, aspect='auto', origin='lower',
            #                cmap='viridis')
            #     plt.title('Predict Spectrogram')
            #     plt.colorbar(format='%+2.0f dB')
            #
            #     plt.subplot(2, 1, 2)
            #     plt.imshow(np.log1p(output.reshape(-1, output.shape[-1])).T, aspect='auto', origin='lower',
            #                cmap='viridis')
            #     plt.title('Output Spectrogram')
            #     plt.colorbar(format='%+2.0f dB')
            #
            #     plt.tight_layout()
            #     plt.show()
        writer.add_scalar('Train Loss', average_loss, epoch)
        print("\n---------------------------------------Train 완료, Validation 시작-------------------------------------------")

        #검증 loop
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for val_batch_x, val_batch_y in val_loader:
                val_batch_x, val_batch_y = val_batch_x.to(device), val_batch_y.to(device)
                val_batch_x = val_batch_x.squeeze(0).float()
                val_batch_y = val_batch_y.squeeze(0).float()

                val_outputs = model(val_batch_x)
                print("------------------------------------------------------------")
                val_loss = criterion(val_outputs, val_batch_y)
                total_val_loss += val_loss.item()

                average_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar('Validation Loss', average_val_loss, epoch)
        print("\n")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_loss:.4f}, Validation Loss: {average_val_loss:.4f}")
        print("\n")

    writer.close()
    torch.save(model.state_dict(), "D:/Py/MR project/model/model_mk2.pth") #모델 저장 위치에 유의!
    print("model_mk2 생성 완료\n")