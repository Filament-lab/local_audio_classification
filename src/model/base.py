from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def build(self):
        raise NotImplementedError
