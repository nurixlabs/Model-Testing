"""
Model Factory for Speech-to-Text Pipeline
Creates model instances based on configuration
"""
import logging

# Import all available models
from models.dolphin_model import DolphinModel
from models.whisper_model import WhisperModel
from models.google_model import GoogleModel
from models.google_model_v2 import GoogleChirp2Model
from models.aws_model import AWSModel
from models.salad_model import SaladModel
from models.deepgram_model import DeepgramModel
from models.sarvam_model import SarvamModel
from models.gladia_model import GladiaModel
from models.azure_model import AzureModel
from models.assemblyai_model import AssemblyAIModel
from models.conformer_marathi import ConformerMarathiModel
from models.nvidia_parakeet import NvidiaParakeetModel
from models.cartesia_whisper_model import CartesiaInkWhisperModel

# Model registry mapping names to classes
MODEL_REGISTRY = {
    'dolphin': DolphinModel,
    'whisper': WhisperModel,
    'google': GoogleModel,
    'google_v2': GoogleChirp2Model,
    'aws': AWSModel,
    'salad': SaladModel,
    'deepgram': DeepgramModel,
    'deepgram_nova3': DeepgramModel,  # Use same class with different config
    'deepgram_nova2': DeepgramModel,  # Use same class with different config
    'sarvam': SarvamModel,
    'gladia': GladiaModel,
    'nvidia_parakeet': NvidiaParakeetModel,
    'assemblyai': AssemblyAIModel,
    'azure': AzureModel,
    'conformer_marathi': ConformerMarathiModel,
    'cartesia': CartesiaInkWhisperModel
}


def get_model(model_name, config):
    """
    Factory function to create model instances based on name.
    
    Args:
        model_name (str): Name of the model to create
        config (dict): Configuration for the model
        
    Returns:
        BaseModel: An instance of the requested model
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_REGISTRY:
        available_models = ', '.join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
    
    logging.info(f"Creating model instance: {model_name}")
    
    try:
        model_class = MODEL_REGISTRY[model_name]
        model_instance = model_class(config)
        
        # Set specific name for Deepgram variants
        if model_name == 'deepgram_nova3':
            model_instance.name = 'deepgram_nova3'
            if not config.get('model'):
                config['model'] = 'nova-3'
        elif model_name == 'deepgram_nova2':
            model_instance.name = 'deepgram_nova2'
            if not config.get('model'):
                config['model'] = 'nova-2'
        
        return model_instance
    except Exception as e:
        logging.error(f"Failed to create model {model_name}: {e}")
        raise


def get_available_models():
    """
    Get list of all available model names.
    
    Returns:
        list: Sorted list of available model names
    """
    return sorted(MODEL_REGISTRY.keys())


def is_model_available(model_name):
    """
    Check if a model is available in the registry.
    
    Args:
        model_name (str): Name of the model to check
        
    Returns:
        bool: True if model is available, False otherwise
    """
    return model_name in MODEL_REGISTRY