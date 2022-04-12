import math
import numpy as np
from src.utils.config_reader import ConfigReader
from src.feature_extraction.base import BaseFeature


class RMS(BaseFeature):
    def __init__(self, config: ConfigReader):
        super().__init__()
        self.cfg = config

    def apply(self, audio_signal: np.ndarray):
        """
        Root mean square energy
        :param  audio_signal: input audio signal after windowing
        :return Root mean square energy
        """
        return [math.sqrt(1 / len(audio_signal) * sum(audio_signal[:] ** 2))]
