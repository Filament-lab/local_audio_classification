import math
import numpy as np
from src.utils.config_reader import ConfigReader
from src.feature_extraction.base import BaseFeature
from src.utils.custom_error_handler import FeatureExtractionException


class MFCC(BaseFeature):
    def __init__(self,  config: ConfigReader):
        """
        Mel Frequency Cepstral Coefficient
        """
        super().__init__()
        self.cfg = config
        self.mfcc_coeff = self.cfg.num_mfcc
        self.total_filters = 40
        self.__init_melfilter()
        self.__init_dct_matrix()

    def __init_dct_matrix(self):
        """
        Init DCT matrix
        """
        self.dct_matrix = self._dctmatrix(self.total_filters, self.mfcc_coeff)

    def __init_melfilter(self):
        """
        Init Mel filter
        """
        self.mel_filter = self._melfilter(self.cfg.sample_rate, self.cfg.fft_size, self.total_filters)

    @staticmethod
    def _melfilter(sampling_rate: int, fft_size: int, total_filters: int):
        """
        Mel filter
        :param  total_filters: number of mel filters
        :return  Mel filter in list
        """
        # Maximum frequency of filter (avoid aliasing)
        maxF = sampling_rate / 2

        # Maximal Mel-frequency
        maxMelF = 2595 * np.log10(1 + maxF / 700)

        # Scatter points in Mel-frequency scale
        melpoints = np.arange(0, (total_filters + 2)) / (total_filters + 1) * maxMelF

        # Convert points in normal frequency scale
        points = 700 * (10 ** (melpoints / 2595) - 1)

        # DTF bins within half fftSize
        DFTbins = np.ceil(points / maxF * (fft_size / 2))

        # Set the first value to 0
        DFTbins[0] = 0

        # Create an empty matrix to store filter
        MelFilter = np.zeros((total_filters, fft_size))

        # Create Triangle filters by each row
        for n in range(0, total_filters):
            low = int(DFTbins[n])         # Triangle start
            center = int(DFTbins[n + 1])  # Top of the Triangle
            high = int(DFTbins[n + 2])    # Triangle end

            UpSlope = center - low        # Number of DFT points in lower side of Triangle
            DownSlope = high - center     # Number of DFT points in upper side of Triangle

            # Create lower side slope
            MelFilter[n, range(low - 1, center)] = np.arange(0, UpSlope + 1) / UpSlope

            # Create upper side slope
            MelFilter[n, range(center - 1, high)] = np.flipud(np.arange(0, DownSlope + 1) / DownSlope)

        return MelFilter

    @staticmethod
    def _dctmatrix(total_filters: int, mfcc_coeff: int) -> np.array:
        """
        DCT matrix
        :param  total_filters: number of mel filters
        :param  mfcc_coeff: number of  coefficients
        """
        # Create an matrix (mfcccoeff * totalfilters)
        [cc, rr] = np.meshgrid(range(0, total_filters), range(0, mfcc_coeff))

        # Calculate DCT
        dct_matrix = np.sqrt(2 / total_filters) * np.cos(math.pi * (2 * cc + 1) * rr / (2 * total_filters))
        dct_matrix[0, :] = dct_matrix[0, :] / np.sqrt(2)
        return dct_matrix

    def apply(self, audio_signal: np.ndarray):
        """
        Main function for Mel Frequency Cepstral Coefficient
        :param  audio_signal Input audio signal as numpy array
        :return mfcc: mfccs in list
        :return mel_fft: mel-scaled fft
        """
        try:
            # Apply FFT and Mel scale filter
            spectrum = abs(np.fft.fft(audio_signal, n=self.cfg.fft_size)) / math.sqrt(self.cfg.fft_size * len(audio_signal))
            mel_fft = np.matmul(self.mel_filter, spectrum)

            # Log scale
            ear_mag = np.log10(mel_fft ** 2, where=mel_fft > 0)

            # Apply DCT to cepstrum
            return list(self.dct_matrix.dot(ear_mag))
        except RuntimeWarning as err:
            raise Exception(err)
        except Exception as err:
            raise FeatureExtractionException(f"Error while extracting MFCC {err}")

    def mel_spectrum(self, input_spectrum: list):
        """
        Make Mel-spectrum from mel-scaled spectrum
        :param  input_spectrum: spectrum in list
        :return : mel-scaled spectrum
        """
        # Apply inverse FFT to mel-scaled spectrum and truncate half
        return list(abs(np.fft.ifft((np.matmul(self.mel_filter, input_spectrum)), self.cfg.fft_size))[:self.cfg.fft_size])
