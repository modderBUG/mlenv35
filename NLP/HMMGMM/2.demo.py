from librosa.feature import mfcc
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile



sampling_freq, audio = librosa.load("input_read.wav")
mfcc_features = mfcc(sampling_freq,audio)
print('\nNumber of windows =', mfcc_features.shape[0])
print('Length of each feature =', mfcc_features.shape[1])

mfcc_features = mfcc_features.T
plt.matshow(mfcc_features)
plt.title('MFCC')
plt.show()