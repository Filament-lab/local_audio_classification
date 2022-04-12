import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from src.utils.config_reader import ConfigReader
from src.feature_extraction.base import BaseFeature
from src.utils.custom_error_handler import FeatureExtractionException


class MelSpectrogram(BaseFeature):
    def __init__(self, config: ConfigReader):
        super().__init__()
        self.cfg = config

    def apply(self,
              audio_data,
              log_scale: bool = False,
              normalize: bool = False,
              visualize: bool = False):
        """
        Extract Mel-spectrogram from one audio
        :param audio_data: Input audio data as numpy array
        :param log_scale: Set to True to log scale
        :param normalize: Set to True to normalize
        :param visualize: Set to True to visualize mel-spectrogram
        """
        try:
            # Extract mel-spectrogram from entire audio file
            mel_spectrogram = librosa.feature.melspectrogram(y=audio_data,
                                                             sr=self.cfg.sample_rate,
                                                             n_fft=self.cfg.fft_size,
                                                             n_mels=self.cfg.num_mels)

            # Convert to log scale
            if log_scale:
                mel_spectrogram = np.log(mel_spectrogram + 1e-9)

            # Normalize
            if normalize is True:
                mel_spectrogram = librosa.util.normalize(mel_spectrogram)

            # Visualize
            if visualize:
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max),
                                         y_axis='mel',
                                         fmax=self.cfg.sample_rate,
                                         x_axis='time')
                plt.colorbar(format='%+2.0f dB')
                plt.show()
            return mel_spectrogram
        except Exception as err:
            raise FeatureExtractionException(f"Error while extracting Mel-spectrogram: {err}")