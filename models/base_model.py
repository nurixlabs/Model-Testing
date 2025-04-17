from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Base class for all transcription models."""
    
    def __init__(self, config):
        """
        Initialize the model with configuration.
        
        Args:
            config (dict): Model-specific configuration
        """
        self.config = config
        self.name = "base"  # Override in subclasses
    
    @abstractmethod
    def load(self):
        """Load the model or initialize API client."""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path):
        """
        Transcribe the audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Dictionary containing:
                - text (str): The transcribed text
                - chunks (list, optional): Word-level information with timing
                - confidence (float, optional): Confidence score
        """
        pass
    
    def __str__(self):
        return f"{self.name} Model"