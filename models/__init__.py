"""
Speech-to-Text model implementations.
Each model provides a consistent interface for transcription through the BaseModel abstract class.
"""

# Import base model and all available models
# from models.base_model import BaseModel
# from models.dolphin_model import DolphinModel
# from models.whisper_model import WhisperModel
# from models.google_model import GoogleModel
# from models.google_chirp2_model import GoogleChirp2Model
# from models.aws_model import AWSModel
# from models.salad_model import SaladModel
# from models.deepgram_model import DeepgramModel
# from models.deepgram_selfhosted_model import DeepgramSelfHostedModel
# from models.sarvam_model import SarvamModel
# from models.gladia_model import GladiaModel
# from models.nvidia_parakeet_model import NvidiaParakeetModel
# from models.assemblyai_model import AssemblyAIModel
# from models.azure_model import AzureModel
# from models.indic_conformer_model import IndicConformerModel
# from models.ai4bharat_conformer_model import AI4BharatConformerModel
from models.conformer_marathi import ConformerMarathiModel

# Import factory function for dynamic model creation
from models.model_factory import get_model, get_available_models, is_model_available

# Public API
__all__ = [
    # Base model
    'BaseModel',
    
    # Individual model classes
    'DolphinModel',
    'WhisperModel',
    'GoogleModel',
    'GoogleChirp2Model',
    'AWSModel',
    'SaladModel',
    'DeepgramModel',
    'DeepgramSelfHostedModel',
    'SarvamModel',
    'GladiaModel',
    'NvidiaParakeetModel',
    'AssemblyAIModel',
    'AzureModel',
    'IndicConformerModel',
    'AI4BharatConformerModel',
    'ConformerMarathiModel',
    # Factory functions
    'get_model',
    'get_available_models',
    'is_model_available',
]