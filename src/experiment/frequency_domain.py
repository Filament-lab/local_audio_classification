import torch
import numpy as np
from src.utils.custom_logger import logger
from src.data_loader.dataset_loader import DatasetLoader
from src.data_loader.torch_dataset_loader import TorchDatasetLoader
from src.utils.config_reader import ConfigReader
from src.pre_process.pre_processor import PreProcess
from src.feature_extraction.feature_selector import FeatureExtractor
from src.model.model_selector import ModelSelector
from sklearn.preprocessing import LabelEncoder


class FrequencyDomain:
    def __init__(self, config: ConfigReader):
        self.config = config
        self.DL = DatasetLoader(self.config)
        self.PR = PreProcess(self.config)
        self.FE = FeatureExtractor(self.config)
        self.MB = ModelSelector(self.config)

    def process_dataset(self):
        """
        1. Load audio and label as dataset
        2. Apply pre-process to audio
        3. Select and extract feature from audio
        4. Train/Test Split
        5. Create DataSet class to use Torch batch loader
        6. Select and build ML model
        """
        # 1.1. Load audio files
        logger.info("Finding data source...")
        audio_file_path_list = self.DL.find_audio_files()

        # 1.2. Load label
        label_df = self.DL.load_label()

        # 1.3. Find audio files that matches to the labels
        self.label_df = self.DL.match_labels(audio_file_path_list, label_df)
        self.classes = self.label_df[self.config.class_column_name].unique()
        self.num_classes = len(self.classes)
        logger.info(f"{self.num_classes} classes found")
        logger.info(f"{self.label_df[self.config.class_column_name].unique()}")
        self.encoder = LabelEncoder()
        self.encoder.fit(self.label_df[self.config.class_column_name].unique())
        self.label_array = self.encoder.transform(self.label_df[self.config.class_column_name].to_numpy())
        unique, counts = np.unique(self.label_array, return_counts=True)
        logger.info(dict(zip(self.encoder.classes_, counts)))

        # 1.4. Load audio files
        logger.info("Loading audio files...")
        audio_array = self.DL.load_audio()

        # 2. Apply pre-process
        logger.info("Applying pre-process...")
        processed_audio_array = self.PR.apply_multiple_with_threading(audio_array)

        # 3. Select and extract feature
        logger.info("Extracting audio feature...")
        self.feature_array = self.FE.extract_feature(processed_audio_array, "mel_spectrogram")

        # 4. Train/Test Split
        logger.info("Creating dataset...")
        dataset_dict = self.DL.train_test_split(self.feature_array, self.label_array)

        # 5. Create dataset loader with TorchDataset format
        TrainDataset = TorchDatasetLoader(dataset_dict["train_label"], dataset_dict["train_data"])
        TestDataset = TorchDatasetLoader(dataset_dict["test_label"], dataset_dict["test_data"])
        self.train_loader = torch.utils.data.DataLoader(TrainDataset, batch_size=128, shuffle=self.config.shuffle)
        self.test_loader = torch.utils.data.DataLoader(TestDataset, batch_size=128, shuffle=self.config.shuffle)

        # 6. Select and build model
        logger.info("Building model...")
        self.MB.select_model("cnn", self.classes)
        self.MB.build()

    def train(self):
        logger.info("Training model...")
        self.MB.train(self.train_loader, self.config.num_epochs)

    def test(self):
        logger.info("Testing model...")
        self.MB.test(test_loader=self.test_loader, show_confusion_matrix=True)


if __name__ == "__main__":
    # Read config file
    CFG = ConfigReader("drums")

    # Initialize experiment class
    FD = FrequencyDomain(CFG)

    # Process dataset
    FD.process_dataset()

    # Train model
    FD.train()

    # Test model
    FD.test()
