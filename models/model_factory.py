from models.dolphin_model import DolphinModel
from models.whisper_model import WhisperModel
from models.google_model import GoogleModel
from models.aws_model import AWSModel
from models.salad_model import SaladModel
from models.deepgram_model import DeepgramModel

def get_model(model_name, config):
    """
    Factory function to create model instances based on name.
    
    Parameters:
        model_name (str): Name of the model to create
        config (dict): Configuration for the model
        
    Returns:
        BaseModel: An instance of the requested model
    """
    model_map = {
        'dolphin': DolphinModel,
        'whisper': WhisperModel,
        'google': GoogleModel,
        'aws': AWSModel,
        'salad': SaladModel,
        'deepgram': DeepgramModel
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(model_map.keys())}")
    
    # Create an instance of the appropriate model class
    return model_map[model_name](config)