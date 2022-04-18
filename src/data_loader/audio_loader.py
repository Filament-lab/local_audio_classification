import os
import glob
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
from src.utils.custom_error_handler import DataLoaderException
# from sklearn.preprocessing import LabelEncoders


class AudioLoader:
    @staticmethod
    def _load_one_audio_file(input_audio_file_path: str, resampling_rate: int=None) -> np.ndarray:
        """
        Load audio file
        :param input_audio_file_path: path to input audio file
        :param resampling_rate: Sample rate to be re-sampled
        :return audio signal as numpy ndarray
        """
        try:
            # Load audio with torchaudio
            audio_data, sampling_rate = torchaudio.load(input_audio_file_path)
            # audio_data_librosa, sampling_rate = librosa.load(input_audio_file_path)

            # Apply re-sampling
            if resampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=resampling_rate)
                audio_data = resampler(audio_data)

            return audio_data
        except Exception as err:
            raise DataLoaderException(
                f"Could not load audio file: {input_audio_file_path}: {err}"
            )

    @staticmethod
    def load_multiple_audio_files(file_path_list: list) -> np.array:
        """
        Get audio chunk from given file path list and return audio as list of numpy array
        :param file_path_list: Target audio file path to load
        :return Audio data as numpy array
        """
        try:
            # Load audio files
            return np.array(
                [
                    AudioLoader._load_one_audio_file(file_path)
                    for file_path in tqdm(file_path_list)
                ], dtype=object
            )
        except Exception as err:
            raise DataLoaderException(f"Error while loading multiple audio files {err}")


if __name__ == "__main__":
    # Load audio files
    audio_data = AudioLoader.load_multiple_audio_files(
        ["file_audio.wav", "another_audio.wav"]
    )
