import os
import pandas as pd
from typing import List
from src.data_loader.label_loader import LabelLoader
from src.data_loader.audio_loader import AudioLoader
from src.utils.config_reader import ConfigReader
from src.utils.file_reader_writer import find_files
from src.utils.custom_logger import logger


class DatasetLoader:
    def __init__(self, config: ConfigReader):
        self.config = config
        # TODO: Validation of the parameters from config

    def load_label(self) -> pd.DataFrame:
        """
        Load label
        :return: Label DataFrame
        """
        return LabelLoader.read_labels_from_csv_file(self.config.label_file_path,
                                                     audio_file_column_name=self.config.audio_column_name,
                                                     class_column_name=self.config.class_column_name,
                                                     sub_class_column_name=self.config.sub_class_column_name)

    def find_audio_files(self) -> List[str]:
        """
        Read audio files under input directory
        :return: list of audio file paths
        """
        return find_files(self.config.audio_folder_path, self.config.audio_file_extension)

    def match_labels(self, audio_file_path_list: List[str], label_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match labels with audio file names
        :param audio_file_path_list: List of audio files
        :param label_df: DataFrame that contains labels for the audio file names
        :return: Filtered DataFrame with only matched items
        """
        # Extract base file names only
        file_names = [os.path.basename(file_path) for file_path in audio_file_path_list]

        # Match audio with labels
        logger.info(f"{len(label_df)} files to be matched")
        matched_df = label_df.loc[label_df[self.config.audio_column_name].isin(file_names)]
        if len(matched_df) == 0:
            logger.error("Failed to match labels with audio file names")
        logger.info(f"{len(matched_df)} audio files matched with labels")
        return matched_df


if __name__ == "__main__":
    # Read config
    CFG = ConfigReader("drums")

    # Initialize dataset class
    DL = DatasetLoader(CFG)

    # Load audio files
    audio_file_path_list = DL.find_audio_files()

    # Load label
    label_df = DL.load_label()

    # Find audio files
    audio_file_path_list = DL.match_labels(audio_file_path_list, label_df)
