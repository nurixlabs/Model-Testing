"""
Speech-to-Text model implementations.
Each model provides a consistent interface for transcription through the BaseModel abstract class.
"""

# Import base model and all available models
from models.base_model import BaseModel
from models.dolphin_model import DolphinModel
from models.whisper_model import WhisperModel
from models.google_model import GoogleModel
from models.google_model_v2 import GoogleChirp2Model
from models.aws_model import AWSModel
from models.salad_model import SaladModel
from models.deepgram_model import DeepgramModel

from models.sarvam_model import SarvamModel
from models.gladia_model import GladiaModel

from models.assemblyai_model import AssemblyAIModel
from models.azure_model import AzureModel
from models.cartesia_whisper_model import CartesiaInkWhisperModel
from models.conformer_marathi import ConformerMarathiModel

# Import factory function for dynamic model creation
from models.model_factory import get_model, get_available_models, is_model_available

# Public API
__all__ = [
    # Base model
    'BaseModel',
    'CartesiaInkWhisperModel',
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