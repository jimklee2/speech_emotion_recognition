import random # random 관련 함수 모듈
import IPython.display as ipd # Audio 출력용 화면

import numpy as np # 배열 함수 모듈
import matplotlib.pyplot as plt # 시각화( 그래프 ) 함수 모듈

# Audio 데이터 처리 함수 모듈
import librosa
import soundfile as sf
import skimage.io



data, sr = librosa.load( './1212.wav', sr = 22050 ) # 음향 데이터 읽기




# mfcc_data = librosa.feature.melspectrogram( data, sr,
#                                             n_fft = 512,
#                                             win_length = 400, 
#                                             hop_length = 160,
#                                             n_mels = 80 )
# print( mfcc_data.shape )

# plt.pcolor( mfcc_data )
# plt.show()

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    mels = np.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)


if __name__ == '__main__':
    # settings
    hop_length = 512 # number of samples per time-step in spectrogram
    n_mels = 128 # number of bins in spectrogram. Height of image
    time_steps = 384 # number of time-steps. Width of image

    # load audio. Using example from librosa
    
    y, sr = librosa.load('./1212.wav', offset=1.0, duration=10.0, sr=22050)
    out = 'out.png'

    # extract a fixed length window
    start_sample = 0 # starting at beginning
    length_samples = time_steps*hop_length
    window = y[start_sample:start_sample+length_samples]
    
    # convert to PNG
    spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
    print('wrote file', out)