from abc import ABC, abstractmethod


class BaseExperiment(ABC):
    """
    Base class for experiment
    All abstractmethod needs to be implemented in the child class
    """
    def __init__(self):
        pass

    @abstractmethod
    def process_dataset(self, *args):
        raise NotImplementedError

    @abstractmethod
    def build_model(self, *args):
        raise NotImplementedError

    @abstractmethod
    def train_model(self, *args):
        raise NotImplementedError

    @abstractmethod
    def test_model(self, *args):
        raise NotImplementedError
