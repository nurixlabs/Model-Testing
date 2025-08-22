"""
Deepgram Speech-to-Text Model Implementation
Supports Nova 2 and Nova 3 models
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
        
        # Model can be nova-2, nova-3, or other Deepgram models
        self.model = config.get('model', 'nova-2')
        
        # Language configuration
        self.language = config.get('language', 'en')
        
        # Additional options
        self.punctuate = config.get('punctuate', True)
        self.smart_format = config.get('smart_format', True)
        self.location = config.get('location', 'us')
        
        # Language mapping for different languages
        self.language_mapping = {
            'english': 'en',
            'en': 'en',
            'en-US': 'en',
            'en-IN': 'en',
            'hindi': 'hi',
            'hi': 'hi',
            'hi-IN': 'hi',
            'marathi': 'mr',
            'mr': 'mr',
            'mr-IN': 'mr',
            'hinglish': 'hi',  # Use Hindi for Hinglish
        }
    
    def load(self):
        """Initialize the Deepgram client."""
        if not self.api_key:
            raise ValueError("Deepgram API key is required. Set it in config or DEEPGRAM_API_KEY environment variable.")
        
        logging.info(f"Initializing Deepgram client")
        logging.info(f"Model: {self.model}, Language: {self.language}, Location: {self.location}")
        
        self.client = DeepgramClient(self.api_key)
        logging.info("Deepgram client initialized successfully")
    
    def _get_language_code(self, language=None):
        """Get the appropriate language code for Deepgram."""
        lang = language or self.language
        return self.language_mapping.get(lang, lang)
    
    def transcribe(self, audio_path, language=None):
        """
        Transcribe audio using Deepgram.
        
        Args:
            audio_path: Path to audio file
            language: Optional language override
            
        Returns:
            dict: Transcription results
        """
        try:
            with open(audio_path, "rb") as audio_file:
                buffer_data = audio_file.read()
            
            payload = {"buffer": buffer_data}
            
            # Get the appropriate language code
            lang_code = self._get_language_code(language)
            
            options = {
                'punctuate': self.punctuate,
                'language': lang_code,
                'model': self.model,
                'smart_format': self.smart_format,
            }
            
            # Only add endpoint if specified
            if self.location:
                options['endpoint'] = self.location
            
            logging.info(f"Transcribing with Deepgram {self.model} in language {lang_code}...")
            
            response = self.client.listen.rest.v("1").transcribe_file(payload, options)
            
            # Access the response object attributes directly
            if not hasattr(response, 'results') or not response.results:
                return {
                    'text': '',
                    'error': 'No results found in response'
                }
            
            results = response.results
            
            # Check if channels exist
            if not hasattr(results, 'channels') or not results.channels:
                return {
                    'text': '',
                    'error': 'No channels found in response'
                }
            
            channels = results.channels
            
            # Get the first channel
            if len(channels) == 0:
                return {
                    'text': '',
                    'error': 'No channels available'
                }
            
            channel = channels[0]
            
            # Check for alternatives
            if not hasattr(channel, 'alternatives') or not channel.alternatives:
                return {
                    'text': '',
                    'error': 'No alternatives found in response'
                }
            
            alternatives = channel.alternatives
            
            if len(alternatives) == 0:
                return {
                    'text': '',
                    'error': 'No alternatives available'
                }
            
            alternative = alternatives[0]
            
            # Extract transcript and confidence
            transcript = getattr(alternative, 'transcript', '')
            confidence = getattr(alternative, 'confidence', 0)
            
            # Extract word timings if available
            words_info = getattr(alternative, 'words', [])
            chunks = []
            
            for word in words_info:
                chunks.append({
                    'word': getattr(word, 'word', ''),
                    'start_time': getattr(word, 'start', 0),
                    'end_time': getattr(word, 'end', 0),
                    'confidence': getattr(word, 'confidence', 0),
                    'punctuated_word': getattr(word, 'punctuated_word', getattr(word, 'word', ''))
                })
            
            return {
                'text': transcript,
                'chunks': chunks,
                'confidence': confidence,
                'model_used': self.model,
                'language_used': lang_code
            }
            
        except Exception as e:
            logging.error(f"Error transcribing with Deepgram {self.model}: {e}")
            return {
                'text': '',
                'error': str(e)
            }