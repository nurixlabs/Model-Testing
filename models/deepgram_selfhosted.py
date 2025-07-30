"""
Deepgram Self-Hosted Model Implementation
For on-premise Deepgram deployments
"""
import os
import requests
import logging
from models.base_model import BaseModel


class DeepgramSelfHostedModel(BaseModel):
    """Deepgram self-hosted speech-to-text implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "deepgram_selfhosted"
        self.api_endpoint = config.get('api_endpoint', 'http://localhost:8080')
        self.api_key = config.get('api_key', '')  # May not be required for self-hosted
        self.model = config.get('model', 'nova-2')
        self.language = config.get('language', 'en')
        self.punctuate = config.get('punctuate', True)
        self.smart_format = config.get('smart_format', True)
    
    def load(self):
        """Initialize the Deepgram self-hosted client."""
        logging.info(f"Initializing Deepgram self-hosted client")
        logging.info(f"Endpoint: {self.api_endpoint}")
        logging.info(f"Model: {self.model}, Language: {self.language}")
        
        # Test connection
        try:
            response = requests.get(f"{self.api_endpoint}/health", timeout=5)
            if response.status_code == 200:
                logging.info("Deepgram self-hosted server is healthy")
            else:
                logging.warning(f"Health check returned status: {response.status_code}")
        except Exception as e:
            logging.warning(f"Could not reach Deepgram self-hosted server: {e}")
    
    def transcribe(self, audio_path):
        """
        Transcribe audio using self-hosted Deepgram.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription results
        """
        try:
            # Read audio file
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Prepare headers
            headers = {
                'Content-Type': 'audio/wav',  # Adjust based on file type
            }
            
            # Add API key if configured
            if self.api_key:
                headers['Authorization'] = f'Token {self.api_key}'
            
            # Prepare query parameters
            params = {
                'model': self.model,
                'language': self.language,
                'punctuate': str(self.punctuate).lower(),
                'smart_format': str(self.smart_format).lower(),
            }
            
            # Make request
            response = requests.post(
                f"{self.api_endpoint}/v1/listen",
                headers=headers,
                params=params,
                data=audio_data,
                timeout=60
            )
            
            if response.status_code != 200:
                logging.error(f"Deepgram API error: {response.status_code}")
                logging.error(f"Response: {response.text}")
                return {
                    'text': '',
                    'error': f'API error: {response.status_code}'
                }
            
            # Parse response
            result = response.json()
            
            # Extract transcript and metadata
            transcript = ''
            chunks = []
            confidence = 0
            
            if 'results' in result:
                channels = result['results'].get('channels', [])
                if channels:
                    alternatives = channels[0].get('alternatives', [])
                    if alternatives:
                        transcript = alternatives[0].get('transcript', '')
                        confidence = alternatives[0].get('confidence', 0)
                        
                        # Extract word timings
                        words = alternatives[0].get('words', [])
                        for word in words:
                            chunks.append({
                                'word': word.get('word', ''),
                                'start_time': word.get('start', 0),
                                'end_time': word.get('end', 0),
                                'confidence': word.get('confidence', confidence),
                                'punctuated_word': word.get('punctuated_word', word.get('word', ''))
                            })
            
            return {
                'text': transcript,
                'chunks': chunks,
                'confidence': confidence
            }
            
        except Exception as e:
            logging.error(f"Error transcribing with Deepgram self-hosted: {e}")
            return {
                'text': '',
                'error': str(e)
            }