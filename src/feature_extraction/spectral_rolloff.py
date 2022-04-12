import math
import numpy as np
from src.utils.config_reader import ConfigReader
from src.feature_extraction.base import BaseFeature


class SpectralRolloff(BaseFeature):
    def __init__(self, config: ConfigReader):
        super().__init__()
        self.cfg = config

    def apply(self, audio_signal: np.ndarray, param: float=0.85):
        """
        Spectral flux
        :param  audio_signal: Audio signal
        :return spectral flux
        """
        # Initialize energy and FFT number
        energy = 0
        count = 0

        # Apply FFT and Mel scale filter
        spectrum = abs(np.fft.fft(audio_signal, n=self.cfg.fft_size)) / math.sqrt(self.cfg.fft_size * len(audio_signal))
        power_spectrum = spectrum[0:int(self.cfg.fft_size/2)]

        # Calculate total energy
        total_energy = sum(power_spectrum[:] ** 2)

        # Find Count which has energy below param*total_energy
        while energy <= param * total_energy and count < len(power_spectrum):
            energy = pow(power_spectrum[count], 2) + energy
            count += 1

        # Normalise Spectral Rolloff
        return [count / len(power_spectrum)]
