import math
import numpy as np
from src.utils.config_reader import ConfigReader
from src.feature_extraction.base import BaseFeature


class SpectralFlux(BaseFeature):
    def __init__(self, config: ConfigReader):
        super().__init__()
        self.previous_power_spectrum = 0
        self.cfg = config

    def apply(self, audio_signal: np.ndarray):
        """
        Spectral flux
        :param  audio_signal: Audio signal
        :return spectral flux
        """
        # Apply FFT and Mel scale filter
        spectrum = abs(np.fft.fft(audio_signal, n=self.cfg.fft_size)) / math.sqrt(self.cfg.fft_size * len(audio_signal))
        power_spectrum = spectrum[0:int(self.cfg.fft_size/2)]

        # Update power spectrum
        flux = math.sqrt((sum(power_spectrum-self.previous_power_spectrum)**2))/(self.cfg.sample_rate/2)
        self.previous_power_spectrum = power_spectrum
        return [flux]


