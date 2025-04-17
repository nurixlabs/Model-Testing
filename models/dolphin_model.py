import os
import re
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
        print(f"Loading Dolphin model '{self.model_size}'...")
        self.model = dolphin.load_model(self.model_size, self.model_dir, self.device)
        print("Dolphin model loaded successfully.")
    
    def clean_hypothesis_text(self, text):
        """
        Remove all tags like <en>, <us>, <asr>, and timestamp tags from hypothesis text.
        """
        # Remove all tags enclosed in < >
        cleaned_text = re.sub(r'<[^>]+>', '', text)
        return cleaned_text.strip()
    
    def transcribe(self, audio_path):
        """
        Transcribe the audio file using Dolphin.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Dictionary containing:
                - text (str): The transcribed text
                - raw_text (str): The raw transcribed text with tags
        """
        try:
            waveform = dolphin.load_audio(audio_path)
            result = self.model(waveform, lang_sym=self.language, region_sym=self.region)
            
            # Get hypothesis from Dolphin result
            hypothesis = result.text.lower()
            
            # Store the raw hypothesis with tags
            raw_hypothesis = hypothesis
            
            # Clean the hypothesis for display/storage
            clean_hypothesis = self.clean_hypothesis_text(hypothesis)
            
            return {
                'raw_text': raw_hypothesis,  # Original text with tags
                'text': clean_hypothesis,    # Cleaned text without tags
            }
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return {'text': "", 'error': str(e)}