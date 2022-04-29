import numpy as np
from src.utils.config_reader import ConfigReader
from src.feature_extraction.mel_spectrogram import MelSpectrogram
from src.feature_extraction.mfcc import MFCC
from src.feature_extraction.spectral_centroid import SpectralCentroid
from src.feature_extraction.spectral_flux import SpectralFlux
from src.feature_extraction.spectral_rolloff import SpectralRolloff
from src.feature_extraction.rms import RMS
from src.feature_extraction.zero_crossing_rate import ZeroCrossingRate
from src.feature_extraction.time_domain_signal import TimeDomainSignal
from src.utils.custom_error_handler import FeatureExtractionException


class FeatureExtractor:
    def __init__(self, config: ConfigReader):
        """
        Feature extraction wrapper
        Add new feature extraction method to "feature_type_map"
        :param config: Config Class
        """
        self.config = config
        self.mel_spectrogram = MelSpectrogram(self.config)
        self.mfcc = MFCC(self.config)
        self.spectral_centroid = SpectralCentroid(self.config)
        self.spectral_flux = SpectralFlux(self.config)
        self.spectral_rolloff = SpectralRolloff(self.config)
        self.rms = RMS(self.config)
        self.zcr = ZeroCrossingRate(self.config)
        self.time_domain_signal = TimeDomainSignal(self.config)

        # Feature extraction selector
        self.feature_type_map = {
            "time_domain_signal": self.time_domain_signal.apply,
            "mel_spectrogram": self.mel_spectrogram.apply,
            "mfcc": self.mfcc.apply,
            "spectral_centroid": self.spectral_centroid.apply,
            "spectral_flux": self.spectral_flux.apply,
            "spectral_rolloff": self.spectral_rolloff.apply,
            "rms": self.rms.apply,
            "zcr": self.zcr.apply
        }

    def select_feature(self, feature_name: str):
        """
        Select audio feature extractor from pre-defined function map
        :param feature_name: Name of feature
        """
        try:
            return self.feature_type_map[feature_name]
        except KeyError:
            raise FeatureExtractionException(f"Selected audio feature extractor '{feature_name}' does not exist")
        except Exception as err:
            raise FeatureExtractionException(f"Error while selecting audio feature extractor: {err}")

    def extract_feature(self, audio_array: np.ndarray, feature_name: str):
        try:
            feature_extractor = self.select_feature(feature_name)
            extracted_feature = feature_extractor(audio_array)
            return extracted_feature
        except Exception as err:
            raise FeatureExtractionException(f"Error while selecting audio feature extractor: {err}")
