import os
import pandas as pd
import numpy as np
from typing import List
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from src.data_loader.label_loader import LabelLoader
from src.data_loader.audio_loader import AudioLoader
from src.utils.config_reader import ConfigReader
from src.utils.file_reader_writer import find_files
from src.utils.custom_logger import logger
from src.utils.multi_process import chunk_list


class DatasetLoader:
    def __init__(self, config: ConfigReader):
        self.config = config
        self.audio_file_path_list = []
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
        file_name_path_dict = {os.path.basename(file_path): str(file_path) for file_path in audio_file_path_list}

        # Match audio with labels
        logger.info(f"{len(label_df)} files to be matched")
        matched_df = label_df.loc[label_df[self.config.audio_column_name].isin(file_names)]

        if len(matched_df) == 0:
            assert "Failed to match labels with audio file names"

        logger.info(f"{len(matched_df)} audio files matched with labels")

        # Update audio file path list with only matched file paths
        for matched_audio_file_name in matched_df["audio_file"]:
            self.audio_file_path_list.append(file_name_path_dict[matched_audio_file_name])
        return matched_df

    def load_audio(self, chunk_size: int = 30) -> List[np.ndarray]:
        """
        Load all audio files into numpy array
        :param chunk_size: Size of one list to load at once
        :return: List of numpy arrays with audio data
        """
        # Load all audio files with multi threading
        chunked_audio_file_path_list = chunk_list(self.audio_file_path_list, chunk_size)
        audio_array = []
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = [pool.apply_async(AudioLoader.load_multiple_audio_files, [chunked_audio_file_path_list[i]]) for i in range(len(chunked_audio_file_path_list))]
            for res in results:
                audio_array.extend(res.get())
        return audio_array

    def train_test_split(self, feature_array: np.ndarray, label_array: np.ndarray) -> dict:
        """
        Take all data and label as input. Split into train/test dataset.
        :param  feature_array: Feature array
        :param  label_array: Label array
        """
        # Split data placeholder
        dataset_dict = {"train_data": None,
                        "train_label": None,
                        "test_data": None,
                        "test_label": None}

        # Split dataset
        train_data, test_data, train_label, test_label = train_test_split(feature_array,
                                                                          label_array,
                                                                          test_size=self.config.test_rate,
                                                                          shuffle=self.config.shuffle,
                                                                          stratify=label_array)

        # Store each split
        dataset_dict["train_data"] = train_data
        dataset_dict["train_label"] = train_label
        dataset_dict["test_data"] = test_data
        dataset_dict["test_label"] = test_label
        return dataset_dict


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
    data_df = DL.match_labels(audio_file_path_list, label_df)

    # Load audio files
    audio_array = DL.load_audio()
