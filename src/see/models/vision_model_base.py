from abc import ABC, abstractmethod

class VisionModel(ABC):
    
    @abstractmethod
    def load(self) -> None:
        pass