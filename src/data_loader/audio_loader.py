import os
import glob
import librosa
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
from src.utils.custom_error_handler import DataLoaderException
from sklearn.preprocessing import LabelEncoder


class AudioLoader:
    @staticmethod
    def _load_one_audio_file(input_audio_file_path: str):
        """
        Load audio file
        :param input_audio_file_path: path to input audio file
        :return audio signal in tuple
        """
        try:
            if str(os.path.split(input_audio_file_path)[1]).lower().endswith(".mp3"):
                int_audio_data_stereo = np.array(
                    AudioSegment.from_mp3(input_audio_file_path).get_array_of_samples()
                )
                int_audio_data_mono = int_audio_data_stereo[
                    :: AudioSegment.from_mp3(input_audio_file_path).channels
                ]
                audio_data = [
                    int_audio_data_mono.astype(np.float32, order="C") / 32768.0
                ]
            # Load mp4
            elif str(os.path.split(input_audio_file_path)[1]).lower().endswith(".mp4"):
                int_audio_data_stereo = np.array(
                    AudioSegment.from_file(
                        input_audio_file_path, "m4a"
                    ).get_array_of_samples()
                )
                int_audio_data_mono = int_audio_data_stereo[
                    :: AudioSegment.from_file(input_audio_file_path).channels
                ]
                audio_data = [
                    int_audio_data_mono.astype(np.float32, order="C") / 32768.0
                ]
            # Load everything else
            else:
                audio_data, sampling_rate = librosa.load(input_audio_file_path)
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
