import os
import time
import requests
from models.base_model import BaseModel

class SaladModel(BaseModel):
    """Salad API speech-to-text implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "salad"
        self.api_key = os.environ.get('SALAD_API_KEY', config.get('api_key'))
        self.organization = config.get('organization', 'nurix-ai')
    
    def load(self):
        """Initialize Salad API."""
        if not self.api_key:
            raise ValueError("Salad API key is required. Set it in config or SALAD_API_KEY environment variable.")
        
        print(f"Salad API initialized for organization {self.organization}")
    
    def _get_presigned_url(self, audio_path):
        """
        For production use, you would upload the file to S3 and generate a presigned URL.
        This is a simplified version that assumes the file is already accessible via URL.
        
        Parameters:
            audio_path (str): Path to the audio file
            
        Returns:
            str: URL to the audio file
        """
        # In a real implementation, you would use boto3 to upload to S3 and generate a presigned URL
        # For this example, let's assume the file is local and we need to handle it differently
        # You would replace this with your actual S3 upload logic
        
        # For demonstration purposes only:
        return f"file://{os.path.abspath(audio_path)}"
    
    def _submit_transcription_job(self, audio_url, audio_file_id):
        """
        Submit a transcription job to the Salad API.
        
        Parameters:
            audio_url (str): URL to the audio file
            audio_file_id (str): ID of the audio file
            
        Returns:
            dict: Response from the Salad API
        """
        url = f"https://api.salad.com/api/public/organizations/{self.organization}/inference-endpoints/transcribe/jobs"
        
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
        
        if response.status_code != 200:
            print(f"API error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        return response.json()
    
    def _get_job_status(self, job_id):
        """
        Get the status of a transcription job.
        
        Parameters:
            job_id (str): Job ID from the submission response
            
        Returns:
            dict: Response from the Salad API
        """
        url = f"https://api.salad.com/api/public/organizations/{self.organization}/inference-endpoints/transcribe/jobs/{job_id}"
        
        headers = {
            "Salad-Api-Key": self.api_key
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"API error checking status: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        return response.json()
    
    def transcribe(self, audio_path):
        """
        Transcribe the audio file using Salad API.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Dictionary containing:
                - text (str): The transcribed text
                - chunks (list): Word-level information with timing
        """
        try:
            # Get file ID from filename
            audio_file_id = os.path.splitext(os.path.basename(audio_path))[0]
            
            # Get a URL for the audio file
            # In a real implementation, you would upload to S3 and generate a presigned URL
            audio_url = self._get_presigned_url(audio_path)
            
            # Submit transcription job
            job_response = self._submit_transcription_job(audio_url, audio_file_id)
            
            if not job_response or 'id' not in job_response:
                return {'text': '', 'error': 'Failed to submit transcription job'}
            
            job_id = job_response['id']
            print(f"Submitted job {job_id} for {audio_file_id}, waiting for completion...")
            
            # Poll for job completion
            max_retries = 60  # 5 minutes at 5-second intervals
            for retry in range(max_retries):
                job_status = self._get_job_status(job_id)
                
                if not job_status:
                    time.sleep(5)
                    continue
                
                status = job_status.get('status', '').lower()
                
                if status in ['succeeded', 'completed']:
                    print(f"Job {job_id} completed!")
                    break
                elif status in ['failed', 'error']:
                    error_message = job_status.get('error', 'Unknown error')
                    return {'text': '', 'error': f"Job failed: {error_message}"}
                
                print(f"Job status: {status}, waiting...")
                time.sleep(5)
            else:
                return {'text': '', 'error': 'Job timed out'}
            
            # Extract transcript from job output
            text = job_status.get('text', '')
            
            # Extract word information
            words_info = []
            if 'words' in job_status:
                words_info = job_status.get('words', [])
            elif 'word_segments' in job_status:
                words_info = job_status.get('word_segments', [])
            
            # Format word timings
            chunks = []
            for word_info in words_info:
                chunks.append({
                    'word': word_info.get('word', ''),
                    'start_time': word_info.get('start', 0),
                    'end_time': word_info.get('end', 0),
                    'confidence': word_info.get('confidence', 0)
                })
            
            return {
                'text': text,
                'chunks': chunks,
                'job_id': job_id
            }
        
        except Exception as e:
            print(f"Error transcribing with Salad API: {e}")
            return {'text': '', 'error': str(e)}