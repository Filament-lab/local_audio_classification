import numpy as np
import librosa
import librosa.display
from src.feature_extraction.base import BaseFeature
from src.utils.custom_error_handler import FeatureExtractionException


class ChromaGram(BaseFeature):
    def __init__(self):
        super().__init__()

    @staticmethod
    def apply(audio_data: np.array, sample_rate: int, bins_per_octave: int=None):
        """
        Extract Mel-spectrogram from one audio
        :param audio_data: Input audio data as numpy array
        :param sample_rate: Sampling Rate
        :param bins_per_octave: Number of bins per octave band
        """
        try:
            # Extract chromagram
            chromagram = librosa.feature.chroma_cqt(y=audio_data, sr=sample_rate, bins_per_octave=bins_per_octave)
            return chromagram
        except Exception:
            try:
                # Extract chromagram
                chromagram = librosa.feature.chroma_cqt(y=audio_data)
                return chromagram
            except Exception as err:
                raise FeatureExtractionException(f"Error while extracting chromagram: {err}")
