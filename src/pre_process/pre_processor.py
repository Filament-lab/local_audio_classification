import math
import numpy as np
from torch import Tensor
from tqdm import tqdm
from typing import List
import multiprocessing as mp
from src.utils.multi_process import chunk_list
from src.utils.augmentation import gaussian_noise, time_shift, time_stretch, pitch_shift
from src.utils.config_reader import ConfigReader
from src.utils.custom_error_handler import PreProcessException


class PreProcess:
    def __init__(self, config: ConfigReader):
        """
        Read parameters from config file
        :param config: ConfigReader
        """
        self.cfg = config
        self.__init__augmentation()

    def __init__augmentation(self):
        """
        Initialize augmentation functions
        """
        self.augmentation_functions = {"gaussian_noise": gaussian_noise,
                                       "time_shift": time_shift,
                                       "time_stretch": time_stretch,
                                       "pitch_shift": pitch_shift}

    @staticmethod
    def normalize_signal(audio_signal: np.ndarray):
        """
        Divide by maximum value of the input array
        :param audio_signal: Input audio signal as numpy array
        """
        try:
            audio_signal = audio_signal / max(abs(audio_signal[:]))
            return audio_signal - np.mean(audio_signal)
        except RuntimeError or RuntimeWarning as err:
            raise PreProcessException(f"Error while normalizing audio signal: {err}")
        except Exception as err:
            raise PreProcessException(f"Error while normalizing audio signal: {err}")

    @staticmethod
    def zero_pad(audio_signal: np.array, max_signal_length: int):
        """
        Zero padding to both sides of input audio signal
        :param audio_signal: Input audio signal as numpy array
        :param max_signal_length: Maximum length of the signal as sample
        """
        try:
            # Calculate pad length
            pad_length = int(max_signal_length - len(audio_signal))
            pad_left = pad_length - int(np.floor(pad_length / 2))
            pad_right = pad_length - pad_left
            return np.pad(audio_signal, [pad_left, pad_right], mode='constant', constant_values=0)
        except Exception as err:
            raise PreProcessException(f"Error while zero padding to audio signal: {err}")

    @staticmethod
    def clip(audio_signal: np.array, max_signal_length: int):
        """
        Clip audio signal
        :param audio_signal: Input audio signal as numpy array
        :param max_signal_length: Maximum length of the signal as sample
        """
        try:
            # Clip audio by defined length
            return audio_signal[:int(max_signal_length)]
        except Exception as err:
            raise PreProcessException(f"Error while zero padding to audio signal: {err}")

    @staticmethod
    def add_eps(audio_data: np.array) -> np.array:
        """
        Add eps to audio data
        :param audio_data: Audio signal as numpy array
        :return Audio signal as numpy array
        """
        return audio_data + np.finfo(float).eps

    @staticmethod
    def sliding_windows(audio_signal: np.array, window_size: int, overlap_ratio: float, normalize_frame: bool) -> np.array:
        """
        Segmentation for the input audio file with the given frame length
        :param audio_signal: audio signal
        :return Framed audio
        """
        # Calculate samples in frame step
        step_samples = math.floor(window_size * overlap_ratio)

        # Calculate number of frames
        num_frames = int(math.floor((len(audio_signal) - step_samples) / step_samples))

        # Store framed audio input into list
        framed_audio_list = []
        for frame in range(num_frames):
            start_index = frame * step_samples
            end_index = start_index + window_size
            audio_frame = audio_signal[start_index:end_index]
            if normalize_frame:
                audio_frame = PreProcess.add_eps(audio_frame)
                normalization_factor = 1 / np.max(np.abs(audio_frame))
                audio_frame = audio_frame * normalization_factor
            framed_audio_list.append(audio_frame)
        return np.array(framed_audio_list)

    @staticmethod
    def apply_window(audio_array: np.array, window_size: int) -> np.array:
        """
        Apply window to the audio signal
        :param audio_array: Input audio array
        :return Windowed audio array
        """
        windowed_array = []
        for array in audio_array:
            windowed_array.append(np.hamming(window_size) * array)
        return np.array(windowed_array)


    def augment_time(self, audio_signal: np.array, time_augmentation_set: set):
        """
        Apply time-domain data augmentation
        :param audio_signal: Time domain audio signal as numpy array
        :return: augmented audio chunk array
        """
        augmented_audio_array = [audio_signal]
        # Apply each augmentation
        if len(time_augmentation_set) >= 1:
            for augmentation in time_augmentation_set:
                augmentation_function = self.augmentation_functions[augmentation]
                augmented_audio = augmentation_function(audio_signal)
                augmented_audio_array.append(augmented_audio)

        return np.array(augmented_audio_array)

    def apply(self,
              audio_signal: np.array,
              zero_padding: bool = True,
              clip: bool = True,
              sliding_window: bool = False,
              apply_window: bool = False,
              time_augmentation_set: set = None):
        # Convert to numpy array
        if isinstance(audio_signal, list) or isinstance(audio_signal, Tensor):
            audio_signal = np.array(audio_signal)

        # Convert n-channel to mono
        if audio_signal.ndim >= 2:
            audio_signal = audio_signal[0]

        # Zero padding if audio signal is longer than maximum seconds
        if zero_padding and len(audio_signal) < self.cfg.max_signal_second * self.cfg.sample_rate:
            audio_signal = PreProcess.zero_pad(audio_signal, self.cfg.max_signal_second * self.cfg.sample_rate)

        # Normalize audio signal
        if self.cfg.normalize_audio:
            audio_signal = PreProcess.add_eps(audio_signal)
            audio_signal = PreProcess.normalize_signal(audio_signal)

        # Augmentation
        audio_signal_list = self.augment_time(audio_signal, time_augmentation_set) if time_augmentation_set else [audio_signal]
        pre_processed_audio_signal_list = []
        try:
            for audio_signal in audio_signal_list:
                # If audio signal is shorter than maximum signal length, apply zero padding to both sides
                if clip and len(audio_signal) > self.cfg.max_signal_second * self.cfg.sample_rate:
                    audio_signal = PreProcess.clip(audio_signal, self.cfg.max_signal_second * self.cfg.sample_rate)

                # Sliding window
                if sliding_window:
                    audio_signal = PreProcess.sliding_windows(audio_signal,
                                                              window_size=self.cfg.window_size,
                                                              overlap_ratio=self.cfg.overlap,
                                                              normalize_frame=self.cfg.normalize_audio_frame)

                # Apply window
                if apply_window:
                    audio_signal = PreProcess.apply_window(audio_signal, window_size=self.cfg.window_size)

                pre_processed_audio_signal_list.append(audio_signal)
            return pre_processed_audio_signal_list

        except Exception as err:
            raise PreProcessException(f"Error while pre-processing {err}")

    def apply_multiple(self, audio_signal_array: List[np.ndarray]) -> List[np.ndarray]:
        # Load all audio files with multi threading
        processed_audio_array = []
        for audio_signal in tqdm(audio_signal_array):
            processed_audio_array.append(self.apply(audio_signal))
        return processed_audio_array

    def apply_multiple_with_threading(self, audio_signal_array: List[np.ndarray], chunk_size: int = 30) -> np.ndarray:
        # Load all audio files with multi threading
        chunked_audio_array = chunk_list(audio_signal_array, chunk_size)
        audio_array = []
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = [pool.apply_async(self.apply_multiple, [chunked_audio_array[i]]) for i in range(len(chunked_audio_array))]
            for res in results:
                audio_array.extend(res.get())
        return np.array(audio_array)
