import os
import numpy as np
import pandas as pd
from typing import Optional
from src.utils.custom_error_handler import DataLoaderException
from sklearn.preprocessing import LabelEncoder


class LabelLoader:
    @staticmethod
    def encode_labels_from_folder_names(
        base_folder_path: str, Encoder: Optional[LabelEncoder]
    ) -> np.array:
        """
        Get labels from given folder names, convert string to numeric label
        :param base_folder_path: Folder path that contains sub-folders as classes
        :param Encoder: Label encoder class
        :return Numeric labels as numpy array
        """
        try:
            # Get labels
            sub_folder_names = os.listdir(base_folder_path)

            # Convert to numeric label
            return (
                Encoder.transform(sub_folder_names)
                if Encoder
                else LabelEncoder().fit_transform(sub_folder_names)
            )
        except Exception as err:
            raise DataLoaderException(f"Error while getting labels {err}")

    @staticmethod
    def read_labels_from_csv_file(
        label_file_path: str,
        audio_file_column_name: str,
        class_column_name: str,
        sub_class_column_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load labels from csv file
        :param label_file_path: csv file path
        :param audio_file_column_name: Column name that contains audio file name
        :param class_column_name: Column name that contains class
        :param sub_class_column_name: Column name that contains sub-class
        :return: Labels as pandas DataFrame
        """
        try:
            # Get labels from file
            label_df = pd.read_csv(label_file_path)

            # Check for outlier labels
            if label_df.isnull().values.any():
                # TODO: Maybe change it to warning
                raise Exception("Label file contains NaN value")

            # Check if required columns exist
            required_column_names = (
                {audio_file_column_name, class_column_name, sub_class_column_name}
                if sub_class_column_name
                else {audio_file_column_name, class_column_name}
            )
            if not required_column_names.issubset(set(label_df.columns)):
                raise Exception(
                    f"'{audio_file_column_name}' and '{class_column_name}' are required column names in the label file"
                )
            return label_df

        except Exception as err:
            raise DataLoaderException(f"Error while loading labels from file: {err}")


if __name__ == "__main__":
    # Load label from csv file
    label_df = LabelLoader.read_labels_from_csv_file(
        label_file_path="datasets/drums/metadata/label.csv",
        audio_file_column_name="audio_file",
        class_column_name="class",
    )
