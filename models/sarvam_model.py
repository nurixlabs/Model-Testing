"""
Sarvam AI Speech-to-Text Model Implementation
"""
import os
import time
import random
import logging
from models.base_model import BaseModel
from sarvamai import SarvamAI


class SarvamModel(BaseModel):
    """Sarvam AI speech-to-text API implementation with rate limiting."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "sarvam"
        self.client = None
        self.api_key = config.get('api_key', os.environ.get('SARVAM_API_KEY'))
        self.model = config.get('model', 'saarika:v2')
        
        # Map language codes to Sarvam's expected format
        raw_language_code = config.get('language_code', 'mr-IN')
        self.language_code = self._map_language_code(raw_language_code)
        
        # Rate limiting configuration
        self.max_retries = config.get('max_retries', 5)
        self.base_delay = config.get('base_delay', 1.0)
        self.max_delay = config.get('max_delay', 60.0)
        self.jitter = config.get('jitter', 0.1)
        self.requests_per_minute = config.get('requests_per_minute', None)
        self.last_request_time = 0
    
    def _map_language_code(self, code):
        """Map common language codes to Sarvam's expected format."""
        # Sarvam accepts: 'unknown', 'hi-IN', 'bn-IN', 'kn-IN', 'ml-IN', 'mr-IN', 
        # 'od-IN', 'pa-IN', 'ta-IN', 'te-IN', 'en-IN', 'gu-IN'
        
        mapping = {
            'en-US': 'en-IN',
            'en': 'en-IN',
            'english': 'en-IN',
            'hi': 'hi-IN',
            'hindi': 'hi-IN',
            'hinglish': 'hi-IN',
            'mr': 'mr-IN',
            'marathi': 'mr-IN',
            'bn': 'bn-IN',
            'bengali': 'bn-IN',
            'kn': 'kn-IN',
            'kannada': 'kn-IN',
            'ml': 'ml-IN',
            'malayalam': 'ml-IN',
            'od': 'od-IN',
            'odia': 'od-IN',
            'pa': 'pa-IN',
            'punjabi': 'pa-IN',
            'ta': 'ta-IN',
            'tamil': 'ta-IN',
            'te': 'te-IN',
            'telugu': 'te-IN',
            'gu': 'gu-IN',
            'gujarati': 'gu-IN'
        }
        
        # If already in correct format, return as is
        if code in ['unknown', 'hi-IN', 'bn-IN', 'kn-IN', 'ml-IN', 'mr-IN', 
                    'od-IN', 'pa-IN', 'ta-IN', 'te-IN', 'en-IN', 'gu-IN']:
            return code
        
        # Map or default to unknown
        return mapping.get(code, 'unknown')
    
    def load(self):
        """Initialize the Sarvam client."""
        if not self.api_key:
            raise ValueError("Sarvam API key is required. Set it in config or SARVAM_API_KEY environment variable.")
        
        logging.info("Initializing Sarvam AI client")
        logging.info(f"Model: {self.model}, Language: {self.language_code}")
        
        self.client = SarvamAI(api_subscription_key=self.api_key)
        logging.info("Sarvam AI client initialized successfully")
    
    def _calculate_delay(self, retry_count):
        """Calculate delay with exponential backoff and jitter."""
        # Exponential backoff with jitter
        delay = min(self.max_delay, self.base_delay * (2 ** retry_count))
        jitter_amount = delay * self.jitter
        delay = delay + random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def _enforce_rate_limit(self):
        """Enforce rate limit by waiting if necessary."""
        if self.requests_per_minute:
            time_between_requests = 60.0 / self.requests_per_minute
            elapsed = time.time() - self.last_request_time
            
            if elapsed < time_between_requests:
                time.sleep(time_between_requests - elapsed)
        
        self.last_request_time = time.time()
    
    def transcribe(self, audio_path):
        """
        Transcribe audio using Sarvam AI with rate limiting and retries.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription results
        """
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # Enforce rate limit
                self._enforce_rate_limit()
                
                with open(audio_path, "rb") as audio_file:
                    response = self.client.speech_to_text.transcribe(
                        file=audio_file,
                        model=self.model,
                        language_code=self.language_code
                    )
                
                # Process response - Sarvam returns an object with transcript attribute
                if hasattr(response, 'transcript'):
                    transcript = response.transcript
                    chunks = []
                    confidence = 0
                    
                    # If timestamps are available, process them
                    if hasattr(response, 'timestamps') and response.timestamps:
                        for timestamp in response.timestamps:
                            chunks.append({
                                'word': timestamp.get('word', ''),
                                'start_time': timestamp.get('start_time', 0),
                                'end_time': timestamp.get('end_time', 0),
                                'confidence': timestamp.get('confidence', 0),
                                'punctuated_word': timestamp.get('word', '')
                            })
                elif isinstance(response, dict):
                    transcript = response.get('transcript', response.get('text', ''))
                    confidence = response.get('confidence', 0)
                    chunks = []
                    
                    raw_chunks = response.get('words', response.get('timestamps', []))
                    for chunk in raw_chunks:
                        chunks.append({
                            'word': chunk.get('word', ''),
                            'start_time': chunk.get('start_time', 0),
                            'end_time': chunk.get('end_time', 0),
                            'confidence': chunk.get('confidence', 0),
                            'punctuated_word': chunk.get('word', '')
                        })
                else:
                    # Handle string response
                    transcript = str(response)
                    chunks = []
                    confidence = 0
                
                return {
                    'text': transcript,
                    'chunks': chunks,
                    'confidence': confidence
                }
                
            except Exception as e:
                error_message = str(e)
                
                # Check for rate limit errors
                if "rate_limit_exceeded" in error_message or "429" in error_message:
                    retry_count += 1
                    
                    if retry_count <= self.max_retries:
                        delay = self._calculate_delay(retry_count - 1)
                        logging.warning(f"Sarvam rate limit exceeded. Retrying in {delay:.2f}s (attempt {retry_count}/{self.max_retries})")
                        time.sleep(delay)
                    else:
                        logging.error(f"Sarvam rate limit exceeded. Maximum retries ({self.max_retries}) reached")
                        return {
                            'text': '',
                            'error': error_message
                        }
                else:
                    # Non-rate-limit errors don't retry
                    logging.error(f"Error transcribing with Sarvam AI: {e}")
                    return {
                        'text': '',
                        'error': error_message
                    }
        
        return {
            'text': '',
            'error': 'Maximum retries reached'
        }