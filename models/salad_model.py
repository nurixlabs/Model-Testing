"""
Salad API Speech-to-Text Model Implementation
"""
import os
import time
import logging
import requests
from models.base_model import BaseModel


class SaladModel(BaseModel):
    """Salad API speech-to-text implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "salad"
        self.api_key = config.get('api_key', os.environ.get('SALAD_API_KEY'))
        self.organization = config.get('organization', 'nurix-ai')
        self.api_base_url = config.get('api_base_url', 'https://api.salad.com')
        self.max_retries = 60
        self.poll_interval = 5
    
    def load(self):
        """Initialize Salad API."""
        if not self.api_key:
            raise ValueError("Salad API key is required. Set it in config or SALAD_API_KEY environment variable.")
        
        logging.info("Initializing Salad API client")
        logging.info(f"Organization: {self.organization}, Base URL: {self.api_base_url}")
        logging.info("Salad API client initialized successfully")
    
    def _get_presigned_url(self, audio_path):
        """
        Get URL for audio file. In production, implement S3 upload and presigned URL generation.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            str: URL to audio file
        """
        logging.warning("Using local file reference. In production, implement S3 upload and presigned URL generation")
        return f"file://{os.path.abspath(audio_path)}"
    
    def _submit_transcription_job(self, audio_url, audio_file_id):
        """Submit transcription job to Salad API."""
        url = f"{self.api_base_url}/api/public/organizations/{self.organization}/inference-endpoints/transcribe/jobs"
        
        headers = {
            "Salad-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": {
                "url": audio_url,
                "language_code": "en",
                "return_as_file": False,
                "sentence_level_timestamps": True,
                "word_level_timestamps": True,
                "diarization": False,
                "sentence_diarization": False,
                "srt": False,
                "summarize": 0,
                "custom_vocabulary": ""
            },
            "metadata": {
                "audio_file_id": audio_file_id
            }
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code not in [200, 201, 202]:
            logging.error(f"Salad API job submission error: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return None
        
        return response.json()
    
    def _get_job_status(self, job_id):
        """Get job status from Salad API."""
        url = f"{self.api_base_url}/api/public/organizations/{self.organization}/inference-endpoints/transcribe/jobs/{job_id}"
        
        headers = {"Salad-Api-Key": self.api_key}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            logging.error(f"Salad API status check error: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return None
        
        return response.json()
    
    def transcribe(self, audio_path):
        """
        Transcribe audio using Salad API.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription results
        """
        try:
            # Get file ID and URL
            audio_file_id = os.path.splitext(os.path.basename(audio_path))[0]
            audio_url = self._get_presigned_url(audio_path)
            
            # Submit transcription job
            job_response = self._submit_transcription_job(audio_url, audio_file_id)
            
            if not job_response or 'id' not in job_response:
                return {
                    'text': '',
                    'error': 'Failed to submit transcription job'
                }
            
            job_id = job_response['id']
            logging.info(f"Submitted Salad job {job_id}, waiting for completion")
            
            # Poll for job completion
            for retry in range(self.max_retries):
                job_status = self._get_job_status(job_id)
                
                if not job_status:
                    time.sleep(self.poll_interval)
                    continue
                
                status = job_status.get('status', '').lower()
                
                if status in ['succeeded', 'completed']:
                    break
                elif status in ['failed', 'error']:
                    error_message = job_status.get('error', 'Unknown error')
                    logging.error(f"Salad job failed: {error_message}")
                    return {
                        'text': '',
                        'error': f"Job failed: {error_message}"
                    }
                
                time.sleep(self.poll_interval)
            else:
                logging.error("Salad job timed out")
                return {
                    'text': '',
                    'error': 'Job timed out'
                }
            
            # Extract transcript and word information
            text = job_status.get('text', '')
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
            logging.error(f"Error transcribing with Salad: {e}")
            return {
                'text': '',
                'error': str(e)
            }