"""
Vision Model Base Class.

Defines the abstract interface that all vision models must implement.
"""

from abc import ABC, abstractmethod


class VisionModel(ABC):
    """
    Abstract base class for vision models.
    
    Enforces that all vision models implement a load() method
    for initialization/model loading. The method behavior depends
    on the model type (some may load from disk, some from network, etc.).
    """
    
    @abstractmethod
    def load(self) -> None:
        """
        Load/initialize the model.
        
        Specific behavior depends on implementation:
        - May load model weights from disk
        - May download from remote server
        - May initialize hardware accelerators
        - May perform pre-processing
        
        Returns:
            None
        """
        pass