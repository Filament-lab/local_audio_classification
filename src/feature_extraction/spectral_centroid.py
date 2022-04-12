import math
import numpy as np
from src.utils.config_reader import ConfigReader
from src.feature_extraction.base import BaseFeature


class SpectralCentroid(BaseFeature):
    def __init__(self, config: ConfigReader):
        super().__init__()
        self.cfg = config

    def apply(self, audio_signal: np.ndarray):
        """
        Spectral centroid
        :param  audio_signal: Audio signal
        :return spectral centroid
        """
        # Apply FFT and Mel scale filter
        spectrum = abs(np.fft.fft(audio_signal, n=self.cfg.fft_size)) / math.sqrt(self.cfg.fft_size * len(audio_signal))
        power_spectrum = spectrum[0:int(self.cfg.fft_size/2)]

        # Calculate frequency bins
        bins = (self.cfg.sample_rate / self.cfg.fft_size) * np.arange(0, int(self.cfg.fft_size / 2))

        # Calculate Spectral Centroid
        centroid = sum(bins * power_spectrum) / sum(power_spectrum)

        # Output normalized spectral centroid
        return [centroid / (self.cfg.sample_rate / 2)]

