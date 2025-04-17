import os
from models.base_model import BaseModel
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

class DeepgramModel(BaseModel):
    """Deepgram speech-to-text API implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "deepgram"
        self.client = None
        self.api_key = os.environ.get('DEEPGRAM_API_KEY', config.get('api_key'))
        self.model = config.get('model', 'nova-3')
        self.language = config.get('language', 'en')
        self.punctuate = config.get('punctuate', True)
        self.smart_format = config.get('smart_format', True)
    
    def load(self):
        """Initialize the Deepgram client."""
        if not self.api_key:
            raise ValueError("Deepgram API key is required. Set it in config or DEEPGRAM_API_KEY environment variable.")
        
        print("Initializing Deepgram client...")
        self.client = DeepgramClient(self.api_key)
        print("Deepgram client initialized successfully.")
    
    def transcribe(self, audio_path):
        """
        Transcribe the audio file using Deepgram.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Dictionary containing:
                - text (str): The transcribed text
                - chunks (list): Word-level information with timing
                - confidence (float): Confidence score
        """
        try:
            with open(audio_path, "rb") as audio_file:
                buffer_data = audio_file.read()
                
            payload = {
                "buffer": buffer_data,
            }
            
            options = {
                'punctuate': self.punctuate,
                'language': self.language,
                'model': self.model,
                'smart_format': self.smart_format,
            }
            
            response = self.client.listen.rest.v("1").transcribe_file(payload, options)
            
            # Extract the transcript
            transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
            
            # Extract word timings
            words_info = response['results']['channels'][0]['alternatives'][0]['words']
            
            # Convert words to a standardized format
            chunks = []
            for word in words_info:
                chunks.append({
                    "word": word.word,
                    "start_time": word.start,
                    "end_time": word.end,
                    "confidence": word.confidence,
                    "punctuated_word": getattr(word, 'punctuated_word', word.word)
                })
            
            # Get confidence score
            confidence = response['results']['channels'][0]['alternatives'][0].get('confidence', 0)
            
            return {
                'text': transcript,
                'chunks': chunks,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Error transcribing with Deepgram: {e}")
            return {'text': '', 'error': str(e)}