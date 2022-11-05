import torch
import numpy as np
from src.experiment.base import BaseExperiment
from src.utils.custom_logger import logger
from src.data_loader.dataset_loader import DatasetLoader
from src.data_loader.torch_dataset_creator import TorchDataset
from src.utils.config_reader import ConfigReader
from src.pre_process.pre_processor import PreProcess
from src.feature_extraction.feature_extractor import FeatureExtractor
from src.model.model_trainer import ModelTrainer
from sklearn.preprocessing import LabelEncoder


class FrequencyDomainMainClassExperiment(BaseExperiment):
    def __init__(self, config: ConfigReader):
        super().__init__()
        self.config = config
        self.DL = DatasetLoader(self.config)
        self.PR = PreProcess(self.config)
        self.FE = FeatureExtractor(self.config)
        self.MT = ModelTrainer(self.config)

    def process_dataset(self):
        """
        1. Load audio and label as dataset
        2. Apply pre-process to audio
        3. Select and extract feature from audio
        4. Train/Test Split
        5. Create DataSet class to use Torch batch loader
        """
        # 1.1. Find all audio files
        logger.info("Finding data source...")
        audio_file_path_list = self.DL.find_audio_files()

        # 1.2. Load label from csv
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
        self.feature_array = self.FE.extract_feature(processed_audio_array, self.config.feature)

        # 4. Train/Test Split
        logger.info("Creating dataset...")
        dataset_dict = self.DL.train_val_test_split(self.feature_array, self.label_array)

        # 5. Create dataset loader with TorchDataset format
        TrainDataset = TorchDataset(dataset_dict["train_label"], dataset_dict["train_data"])
        ValidationDataset = TorchDataset(dataset_dict["validation_label"], dataset_dict["validation_data"])
        TestDataset = TorchDataset(dataset_dict["test_label"], dataset_dict["test_data"])
        self.train_loader = torch.utils.data.DataLoader(TrainDataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle)
        self.validation_loader = torch.utils.data.DataLoader(ValidationDataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle)
        self.test_loader = torch.utils.data.DataLoader(TestDataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle)

    def build_model(self):
        """
        Select and build ML model
        """
        logger.info("Building model...")
        self.MT.select_model(self.config.model_name, self.classes)
        self.MT.build()

    def train_model(self):
        """
        Train model
        """
        logger.info("Training model...")
        self.MT.train(self.train_loader, self.validation_loader, self.config.num_epochs, visualize=True)

    def test_model(self):
        """
        Test model
        :return:
        """
        logger.info("Testing model...")
        self.MT.test(test_loader=self.test_loader, show_confusion_matrix=True)


if __name__ == "__main__":
    # Read config file
    CFG = ConfigReader("mel_spectrogram")

    # Initialize experiment class
    Experiment = FrequencyDomainMainClassExperiment(CFG)

    # Process dataset
    Experiment.process_dataset()

    # Build model
    Experiment.build_model()

    # Train model
    Experiment.train_model()

    # Test model
    Experiment.test_model()