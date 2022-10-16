from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError