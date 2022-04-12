from src.utils.config_reader import ConfigReader
from src.feature_extraction.base import BaseFeature
from src.utils.custom_error_handler import FeatureExtractionException


class TimeDomainSignal(BaseFeature):
    def __init__(self, config: ConfigReader):
        super().__init__()
        self.cfg = config

    def apply(self, audio_data):
        """
        Extract time domain audio signal
        :param audio_data: Input audio data as numpy array
        """
        try:
            # Return time domain audio signal
            return audio_data
        except Exception as err:
            raise FeatureExtractionException(f"Error while extracting time domain audio signal: {err}")
