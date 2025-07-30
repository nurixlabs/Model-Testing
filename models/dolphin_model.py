"""
Dolphin Speech-to-Text Model Implementation
"""
import os
import re
import logging
import dolphin
from models.base_model import BaseModel


class DolphinModel(BaseModel):
    """Dolphin speech-to-text model implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "dolphin"
        self.model = None
        self.model_size = config.get('model_size', 'small')
        self.model_dir = os.path.expanduser(config.get('model_dir', '~/.cache/dolphin'))
        self.device = config.get('device', 'cuda')
        self.language = config.get('language', 'en')
        self.region = config.get('region', 'US')
    
    def load(self):
        """Load the Dolphin model."""
        logging.info(f"Loading Dolphin model '{self.model_size}'")
        logging.info(f"Model directory: {self.model_dir}, Device: {self.device}")
        logging.info(f"Language: {self.language}, Region: {self.region}")
        
        try:
            self.model = dolphin.load_model(self.model_size, self.model_dir, self.device)
            logging.info("Dolphin model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load Dolphin model: {e}")
            raise
    
    def clean_hypothesis_text(self, text):
        """
        Remove all tags like <en>, <us>, <asr>, and timestamp tags from hypothesis text.
        
        Args:
            text: Raw text with tags
            
        Returns:
            str: Cleaned text without tags
        """
        # Remove all tags enclosed in < >
        cleaned_text = re.sub(r'<[^>]+>', '', text)
        return cleaned_text.strip()
    
    def transcribe(self, audio_path):
        """
        Transcribe audio using Dolphin.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription results
        """
        try:
            waveform = dolphin.load_audio(audio_path)
            result = self.model(waveform, lang_sym=self.language, region_sym=self.region)
            
            # Get hypothesis from Dolphin result
            hypothesis = result.text.lower()
            
            # Store raw hypothesis with tags and clean version
            raw_hypothesis = hypothesis
            clean_hypothesis = self.clean_hypothesis_text(hypothesis)
            
            return {
                'text': clean_hypothesis,      # Cleaned text without tags
                'raw_text': raw_hypothesis,    # Original text with tags
                'chunks': [],                  # Dolphin doesn't provide word-level timing
                'confidence': 0                # Dolphin doesn't provide confidence scores
            }
            
        except Exception as e:
            logging.error(f"Error transcribing with Dolphin: {e}")
            return {
                'text': '',
                'error': str(e)
            }