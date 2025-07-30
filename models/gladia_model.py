"""
Gladia Speech-to-Text Model Implementation
"""
import os
import time
import logging
import requests
from models.base_model import BaseModel


class GladiaModel(BaseModel):
    """Gladia API speech-to-text implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "gladia"
        self.api_key = config.get('api_key', os.environ.get('GLADIA_API_KEY'))
        self.model = config.get('model', 'solaria-1')
        self.target_languages = config.get('target_languages', ['en'])
        self.max_retries = 60
        self.poll_interval = 5
    
    def load(self):
        """Initialize the Gladia client."""
        if not self.api_key:
            raise ValueError("Gladia API key is required. Set it in config or GLADIA_API_KEY environment variable.")
        
        logging.info("Initializing Gladia API client")
        logging.info(f"Model: {self.model}, Target languages: {self.target_languages}")
        logging.info("Gladia API client initialized successfully")
    
    def _submit_transcription_job(self, audio_url):
        """Submit transcription job to Gladia API."""
        url = "https://api.gladia.io/v2/pre-recorded"
        headers = {
            "x-gladia-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "audio_url": audio_url,
            "model": self.model,
            "target_languages": self.target_languages
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            logging.error(f"Gladia job submission error: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return None, None
        
        data = response.json()
        return data.get("id"), data.get("result_url")
    
    def _get_job_status(self, result_url):
        """Get job status from Gladia API."""
        headers = {"x-gladia-key": self.api_key}
        response = requests.get(result_url, headers=headers)
        
        if response.status_code != 200:
            logging.error(f"Gladia status check error: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return None
        
        return response.json()
    
    def transcribe(self, audio_url):
        """
        Transcribe audio using Gladia API.
        
        Args:
            audio_url: URL to audio file (Gladia works with URLs, not local files)
            
        Returns:
            dict: Transcription results
        """
        try:
            # Submit transcription job
            job_id, result_url = self._submit_transcription_job(audio_url)
            if not job_id or not result_url:
                return {
                    'text': '', 
                    'error': 'Failed to submit transcription job'
                }
            
            logging.info(f"Submitted Gladia job {job_id}, waiting for completion")
            
            # Poll for completion
            for retry in range(self.max_retries):
                job_status = self._get_job_status(result_url)
                if not job_status:
                    time.sleep(self.poll_interval)
                    continue
                
                status = job_status.get('status', '').lower()
                
                if status in ['done', 'succeeded', 'completed']:
                    break
                elif status in ['failed', 'error']:
                    error_message = job_status.get('error', 'Unknown error')
                    logging.error(f"Gladia job failed: {error_message}")
                    return {
                        'text': '',
                        'error': f"Job failed: {error_message}"
                    }
                
                time.sleep(self.poll_interval)
            else:
                logging.error("Gladia job timed out")
                return {
                    'text': '',
                    'error': 'Job timed out'
                }
            
            # Extract transcription text
            text = job_status.get('transcription', '') or job_status.get('text', '')
            
            # Handle nested result structure
            if not text:
                result = job_status.get('result', {})
                transcription = result.get('transcription', {})
                text = transcription.get('full_transcript', '')
                
                if not text and 'utterances' in transcription and transcription['utterances']:
                    text = transcription['utterances'][0].get('text', '')
            
            # Extract word information
            words_info = job_status.get('words', []) or job_status.get('word_segments', [])
            chunks = []
            
            for word_info in words_info:
                chunks.append({
                    'word': word_info.get('word', ''),
                    'start_time': word_info.get('start', 0),
                    'end_time': word_info.get('end', 0),
                    'confidence': word_info.get('confidence', 0),
                    'punctuated_word': word_info.get('word', '')
                })
            
            # Calculate average confidence
            confidence = sum(chunk.get('confidence', 0) for chunk in chunks) / len(chunks) if chunks else 0
            
            return {
                'text': text,
                'chunks': chunks,
                'confidence': confidence
            }
            
        except Exception as e:
            logging.error(f"Error transcribing with Gladia: {e}")
            return {
                'text': '',
                'error': str(e)
            }