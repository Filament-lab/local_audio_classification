import os
import configparser


class ConfigException(Exception):
    pass


class ConfigReader:
    def __init__(self, experiment: str):
        """
        Read common configuration data from configuration file
        :param experiment: Name of experiment e.g. drums
        """
        # Initialize config parser class
        cfg = configparser.ConfigParser()
        self.cfg = cfg

        # Check if config file exists
        config_file_path = f"config/{experiment}.ini"
        if not os.path.isfile(config_file_path):
            raise ConfigException(f"Config file does not exist in given file path: {config_file_path}")

        # Read module specific config reader
        cfg.read(config_file_path)
        self.__init_dataset(cfg)
        self.__init_pre_process(cfg)
        self.__init_feature_extraction(cfg)
        self.__init_classifier(cfg)

    def __init_dataset(self, cfg: configparser.ConfigParser):
        """
        Extract parameters from dataset field in config file
        :param cfg: Config Reader class
        """
        try:
            self.audio_folder_path = str(cfg.get("dataset", "audio_folder_path"))
            self.audio_file_extension = str(cfg.get("dataset", "audio_file_extension"))
            self.label_file_path = str(cfg.get("dataset", "label_file_path"))
            self.audio_column_name = str(cfg.get("dataset", "audio_column_name"))
            self.class_column_name = str(cfg.get("dataset", "class_column_name"))
            self.sub_class_column_name = str(cfg.get("dataset", "sub_class_column_name"))
            self.test_rate = float(cfg.get("dataset", "test_rate"))
            self.validation_rate = float(cfg.get("dataset", "validation_rate"))
            self.batch_size = int(cfg.get("dataset", "batch_size"))
            self.shuffle = bool(cfg.getboolean("dataset", "shuffle"))
        except Exception as err:
            raise ConfigException(f"Error while reading parameter for dataset: {err}")

    def __init_pre_process(self, cfg: configparser.ConfigParser):
        """
        Extract parameters from pre-process field in config file
        :param cfg: Config Reader class
        """
        try:
            self.normalize_audio = bool(cfg.getboolean("pre_process", "normalize_audio"))
            self.normalize_audio_frame = bool(cfg.getboolean("pre_process", "normalize_audio_frame"))
            self.max_signal_second = float(cfg.get("pre_process", "max_signal_second"))
            self.window_type = str(cfg.get("pre_process", "window_type"))
            self.window_size = int(cfg.get("pre_process", "window_size"))
            self.overlap = float(cfg.get("pre_process", "overlap"))
        except Exception as err:
            raise ConfigException(f"Error while reading parameter for pre-process: {err}")

    def __init_feature_extraction(self, cfg: configparser.ConfigParser):
        """
        Extract parameters from feature extraction field in config file
        :param cfg: Config Reader class
        """
        try:
            self.num_mels = int(cfg.get("feature_extraction", "num_mels"))
            self.num_mfcc = int(cfg.get("feature_extraction", "num_mfcc"))
            self.sample_rate = int(cfg.get("feature_extraction", "sample_rate"))
            self.fft_size = int(cfg.get("feature_extraction", "fft_size"))
            self.feature = str(cfg.get("feature_extraction", "feature"))

        except Exception as err:
            raise ConfigException(f"Error while reading parameter for feature extraction: {err}")

    def __init_classifier(self, cfg: configparser.ConfigParser):
        """
        Extract parameters from classifier field in config file
        :param cfg: Config Reader class
        """
        try:
            self.model_name = str(cfg.get("classifier", "model_name"))
            self.num_epochs = int(cfg.get("classifier", "num_epochs"))
            self.learning_rate = float(cfg.get("classifier", "learning_rate"))
            self.model_folder_path = str(cfg.get("classifier", "model_folder_path"))
        except Exception as err:
            raise ConfigException(f"Error while reading parameter for classifier: {err}")
