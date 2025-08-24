"""
NVIDIA Parakeet-TDT NeMo Model Implementation
"""
import os
import logging
import torch
from models.base_model import BaseModel

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    nemo_asr = None


class NvidiaParakeetModel(BaseModel):
    """NVIDIA Parakeet-TDT 0.6B v2 speech-to-text model using NeMo Toolkit."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "nvidia_parakeet"
        self.model = None
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_id = config.get('model_id', 'nvidia/parakeet-tdt-0.6b-v2')
    
    def load(self):
        """Load the NeMo ASR model."""
        if not NEMO_AVAILABLE:
            raise ImportError("NeMo toolkit is not available")
            
        logging.info(f"Loading NeMo Parakeet model: {self.model_id}")
        logging.info(f"Device: {self.device}")
        
        try:
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_id)
            logging.info("NeMo Parakeet model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load NeMo Parakeet model: {e}")
            raise
    
    def transcribe(self, audio_path):
        """
        Transcribe audio using NVIDIA Parakeet NeMo.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription results
        """
        if not os.path.exists(audio_path):
            return {
                'text': '',
                'error': f'Audio file not found: {audio_path}'
            }
        
        try:
            # Transcribe with timestamps enabled
            output = self.model.transcribe([audio_path], timestamps=True)
            
            if not output or not output[0]:
                return {
                    'text': '',
                    'error': 'No transcription output received'
                }
            
            result = output[0]
            transcript_text = result.text or ''
            
            # Extract word-level timing information
            chunks = []
            if hasattr(result, 'timestamp') and result.timestamp:
                words = result.timestamp.get('word', [])
                for word in words:
                    chunks.append({
                        'word': word.get('word', ''),
                        'start_time': word.get('start', 0),
                        'end_time': word.get('end', 0),
                        'confidence': 0,  # NeMo doesn't provide word-level confidence
                        'punctuated_word': word.get('word', '')
                    })
            
            return {
                'text': transcript_text,
                'chunks': chunks,
                'confidence': 0  # NeMo doesn't provide overall confidence scores
            }
            
        except Exception as e:
            logging.error(f"Error transcribing with NVIDIA Parakeet: {e}")
            return {
                'text': '',
                'error': str(e)
            }