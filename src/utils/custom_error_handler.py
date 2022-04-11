class DataPipelineException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class DataLoaderException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class ConfigException(Exception):
    def __init__(self, message):
        self.message = message


class PreProcessException(Exception):
    def __init__(self, message):
        self.message = message


class FeatureExtractionException(Exception):
    def __init__(self, message):
        self.message = message


class DSPModelException(Exception):
    def __init__(self, message):
        self.message = message


class MLModelException(Exception):
    def __init__(self, message):
        self.message = message


class DataFormatException(Exception):
    def __init__(self, message):
        self.message = message
