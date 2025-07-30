"""
Deepgram Speech-to-Text Model Implementation
"""
import os
import logging
from models.base_model import BaseModel
from deepgram import DeepgramClient, PrerecordedOptions, FileSource


class DeepgramModel(BaseModel):
    """Deepgram speech-to-text API implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "deepgram"
        self.client = None
        self.api_key = config.get('api_key', os.environ.get('DEEPGRAM_API_KEY'))
        self.model = config.get('model', 'nova-3')
        self.language = config.get('language', 'en')
        self.punctuate = config.get('punctuate', True)
        self.smart_format = config.get('smart_format', True)
        self.location = config.get('location', 'us')
    
    def load(self):
        """Initialize the Deepgram client."""
        if not self.api_key:
            raise ValueError("Deepgram API key is required. Set it in config or DEEPGRAM_API_KEY environment variable.")
        
        logging.info("Initializing Deepgram client")
        logging.info(f"Model: {self.model}, Language: {self.language}, Location: {self.location}")
        
        self.client = DeepgramClient(self.api_key)
        logging.info("Deepgram client initialized successfully")
    
    def transcribe(self, audio_path):
        """
        Transcribe audio using Deepgram.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription results
        """
        try:
            with open(audio_path, "rb") as audio_file:
                buffer_data = audio_file.read()
            
            payload = {"buffer": buffer_data}
            
            options = {
                'punctuate': self.punctuate,
                'language': self.language,
                'model': self.model,
                'smart_format': self.smart_format,
                'endpoint': self.location
            }
            
            response = self.client.listen.rest.v("1").transcribe_file(payload, options)
            
            # Extract transcript and metadata
            results = getattr(response, 'results', {})
            channels = results.get('channels', [])
            
            if not channels:
                return {
                    'text': '',
                    'error': 'No channels found in response'
                }
            
            alternatives = channels[0].get('alternatives', [])
            if not alternatives:
                return {
                    'text': '',
                    'error': 'No alternatives found in response'
                }
            
            alternative = alternatives[0]
            transcript = alternative.get('transcript', '')
            confidence = alternative.get('confidence', 0)
            
            # Extract word timings
            words_info = alternative.get('words', [])
            chunks = []
            
            for word in words_info:
                chunks.append({
                    'word': word.get('word', ''),
                    'start_time': word.get('start', 0),
                    'end_time': word.get('end', 0),
                    'confidence': word.get('confidence', 0),
                    'punctuated_word': word.get('punctuated_word', word.get('word', ''))
                })
            
            return {
                'text': transcript,
                'chunks': chunks,
                'confidence': confidence
            }
            
        except Exception as e:
            logging.error(f"Error transcribing with Deepgram: {e}")
            return {
                'text': '',
                'error': str(e)
            }