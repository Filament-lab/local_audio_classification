import json
import pickle
import requests
import librosa
import numpy as np
import random
from sklearn.model_selection import train_test_split
from src.utils.custom_error_handler import DataFormatException


def normalize_array(array: np.array,
                    array_mean: float = None,
                    array_std: float = None,
                    array_max: float = None,
                    array_min: float = None):
    """
    Normalize the input numpy array from -1 to 1
    :param array: Frequency domain labels
    :param array_mean: Mean value (if pre-defined)
    :param array_std: Std value (if pre-defined)
    :param array_max: Max value (if pre-defined)
    :param array_min: Min value (if pre-defined)
    :return: Normalized labels
    """
    # Calculate mean and standard deviation if not given
    if not array_mean and not array_std:
        array_mean, array_std = np.mean(array), np.std(array)
    array = (array - array_mean) / array_std

    # Calculate max and min if not given
    if not array_max and not array_min:
        array_min = np.min(array)
        array_max = np.max(array)
    return 2 * (array - array_min) / (array_max - array_min) - 1


def custom_train_test_split(feature_array: np.array, test_size: float, label_array: np.array = None):
    """
    Take all data and label as input. Split into train/test dataset.
    :param  feature_array: extracted feature in 2D numpy array or 3D numpy array
    :param test_size: Test size 0 < x < 1
    :param label_array: Labels as numpy array
    """
    # Split placeholder
    dataset_dict = {"training_annotation": None,
                    "test_annotation": None,
                    "validation_annotation": None}

    # Split dataset
    if label_array is not None:
        training_annotations, validation_annotations, _, _ = train_test_split(feature_array,
                                                                              label_array,
                                                                              test_size=test_size,
                                                                              stratify=label_array)
    else:
        training_annotations, validation_annotations = train_test_split(feature_array, test_size=test_size)

    # Store each split
    dataset_dict["training_annotation"] = training_annotations
    dataset_dict["test_annotation"] = validation_annotations
    dataset_dict["validation_annotation"] = validation_annotations
    return dataset_dict


def binary_to_float(audio_binary: bytearray) -> np.array:
    """
    Convert binary audio data into floating numpy array
    Args:
        audio (np.array): Audio binary data

    Returns:
        (np.array): Audio array
    """
    return pickle.loads(audio_binary)


def dict2json(output_file_path: str, input_dictionary: dict):
    with open(output_file_path, 'w') as fh:
        json.dump(input_dictionary, fh)


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


def custom_noise(audio_array: np.array, noise_sha_list: list):
    """
    Add custom noise randomly picked from the given noise list
    :param audio_array: Input audio array
    :param noise_sha_list: List of sha
    :return: Audio signal with custom noise added
    """
    # Random factor and select sha
    random_factor = np.random.uniform(0.01, 0.1)
    selected_sha = random.sample(noise_sha_list, 1)[0]

    # Randomly select sha
    noise_audio_chunk = load_audio(host, selected_sha)
    noise_audio_chunk = normalize_signal(noise_audio_chunk)

    if len(noise_audio_chunk) >= len(audio_array):
        noise_audio_chunk = noise_audio_chunk[:len(audio_array)]
    else:
        noise_audio_chunk = zero_pad(noise_audio_chunk, len(audio_array))

    # Add noise
    audio_array = audio_array + noise_audio_chunk * random_factor
    return audio_array / max(abs(audio_array[:]))

