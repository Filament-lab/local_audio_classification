import numpy as np
from src.utils.config_reader import ConfigReader
from src.feature_extraction.base import BaseFeature


class ZeroCrossingRate(BaseFeature):
    def __init__(self, config: ConfigReader):
        super().__init__()
        self.cfg = config

    def apply(self, audio_signal: np.ndarray):
        """
        Zero Crossing Rate
        :param  audio_signal: input audio signal after windowing
        :return zero crossing rate
        """
        # Size of windowed signal
        window_size = len(audio_signal)

        # Slided signal
        xw2 = np.zeros(window_size)
        xw2[1:] = audio_signal[0:-1]

        # Compute Zero-crossing Rate
        return [(1/(2*window_size)) * sum(abs(np.sign(audio_signal)-np.sign(xw2)))]
