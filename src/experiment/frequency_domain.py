import torch
from torch.functional import F
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
        2. Extract feature from audio
        3. Train/Test Split
        """
        # 1.1. Load audio files
        audio_file_path_list = self.DL.find_audio_files()

        # 1.2. Load label
        label_df = self.DL.load_label()

        # 1.3. Find audio files that matches to the labels
        self.label_df = self.DL.match_labels(audio_file_path_list, label_df)
        self.label_array = LabelEncoder().fit_transform(self.label_df[self.config.class_column_name].to_numpy())

        # 1.4. Load audio files
        audio_array = self.DL.load_audio()

        # 2. Apply pre-process
        processed_audio_array = self.PR.apply_multiple_with_threading(audio_array)

        # 3. Extract feature
        self.feature_array = self.FE.extract_feature(processed_audio_array, "mel_spectrogram")

        # 4. Train/Test Split
        dataset_dict = self.DL.train_test_split(self.feature_array, self.label_array)

        # 5. Create dataset loader with TorchDataset format
        TrainDataset = TorchDatasetLoader(dataset_dict["train_label"], dataset_dict["train_data"])
        TestDataset = TorchDatasetLoader(dataset_dict["train_label"], dataset_dict["train_data"])
        self.train_loader = torch.utils.data.DataLoader(TrainDataset, batch_size=128, shuffle=self.config.shuffle)
        self.test_loader = torch.utils.data.DataLoader(TestDataset, batch_size=128, shuffle=self.config.shuffle)

        # 6. Select and Build model
        self.MB.select_model("cnn")
        self.MB.build()

    def train(self):
        self.MB.train(self.train_loader, self.config.num_epochs)

    def test(self):
        self.MB.test(self.test_loader)


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
