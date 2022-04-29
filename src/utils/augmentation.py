import json
import pickle
import librosa
import numpy as np
import random
from sklearn.model_selection import train_test_split


def gaussian_noise(audio_array: np.array, randomize_factor: float = None) -> np.array:
    """
    Add Gaussian noise to audio array
    :param audio_array: Input audio numpy array
    :param randomize_factor: Randomize factor for noise variance
    :return: Audio array with gaussian noise
    """
    if not randomize_factor:
        randomize_factor = np.random.uniform(0.001, 0.1)
    return audio_array + randomize_factor * np.random.normal(0, 1, audio_array.shape)


def time_shift(audio_array: np.array, sampling_rate: int = 8000, shift_factor: int = None) -> np.array:
    """
    Time shift audio array by sampling rate/shift factor
    :param audio_array: Input audio array
    :param sampling_rate: Sampling rate
    :param shift_factor: Shift factor
    :return: Time shifted audio array
    """
    if not shift_factor:
        shift_factor = np.random.randint(1, 10)
    return np.roll(audio_array, int(sampling_rate/shift_factor))


def time_stretch(audio_array: np.array, stretch_factor: float = None) -> np.array:
    """
    Time stretch audio array by stretch factor
    :param audio_array: Input audio array
    :param stretch_factor: Stretch factor 0 < x < 1.0
    :return: Time stretched audio array
    """
    if not stretch_factor:
        stretch_factor = np.random.uniform(0.8, 1.0)
    return librosa.effects.time_stretch(audio_array, stretch_factor)


def pitch_shift(audio_array: np.array, sampling_rate: int = 8000, shift_factor: float = None) -> np.array:
    """
    Pitch shift audio array by shift factor
    :param audio_array: Input audio array
    :param sampling_rate: Sampling rate
    :param shift_factor: Pitch shift factor -5.0 < x < 5.0
    :return: Pitch shifted audio array
    """
    if not shift_factor:
        shift_factor = np.random.uniform(-4.0, 4.0)
    return librosa.effects.pitch_shift(audio_array, sampling_rate, n_steps=shift_factor)


def zero_pad(audio_signal: np.array, max_signal_length: int):
    """
    Zero padding to both sides of input audio signal
    :param audio_signal: Input audio signal as numpy array
    :param max_signal_length: Maximum length of the signal as sample
    """
    # Calculate pad length
    pad_length = int(max_signal_length - len(audio_signal))
    pad_left = pad_length - int(np.floor(pad_length / 2))
    pad_right = pad_length - pad_left
    return np.pad(audio_signal, [pad_left, pad_right], mode='constant', constant_values=0)


def normalize_signal(audio_signal: np.array):
    """
    Divide by maximum value of the input array
    :param audio_signal: Input audio signal as numpy array
    """
    try:
        audio_signal = audio_signal / max(abs(audio_signal[:]))
        return audio_signal - np.mean(audio_signal)
    except ZeroDivisionError:
        audio_signal = audio_signal + np.finfo(float).eps
        return audio_signal - np.mean(audio_signal)

