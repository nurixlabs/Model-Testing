"""
Google Speech-to-Text v1 API Model Implementation
"""
import os
import base64
import logging
import requests
import subprocess
from models.base_model import BaseModel
from pathlib import Path


class GoogleModel(BaseModel):
    """Google Speech-to-Text v1 API implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "google"
        self.api_key = config.get('api_key', os.environ.get('GOOGLE_API_KEY'))
        self.language_code = config.get('language_code', 'en-US')
        self.project_id = config.get('project_id', os.environ.get('GOOGLE_PROJECT_ID', 'acoustic-shade-453507-f6'))
        self.gcloud_path = config.get('gcloud_path', './google-cloud-sdk/bin/gcloud')
    
    def load(self):
        """Initialize Google Speech-to-Text setup."""
        logging.info("Initializing Google Speech-to-Text v1")
        logging.info(f"Language: {self.language_code}, Project ID: {self.project_id}")
        logging.info(f"gcloud path: {self.gcloud_path}")
        
        if not self.api_key and not self._check_gcloud_auth():
            logging.warning("No Google API key provided and gcloud authentication not set up")
            logging.warning("You will need to authenticate with Google Cloud before transcription")
        else:
            logging.info("Google Speech-to-Text initialized successfully")
        
        if not self.project_id:
            logging.warning("No Google Project ID provided for quota project")
            logging.warning("You may encounter permission errors. Set GOOGLE_PROJECT_ID environment variable")
    
    def _check_gcloud_auth(self):
        """Check if gcloud authentication is set up."""
        try:
            result = subprocess.run(
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logging.debug(f"gcloud auth check failed: {result.stderr}")
                return False
                
            return bool(result.stdout.strip())
        except Exception as e:
            logging.debug(f"Error checking gcloud auth: {e}")
            return False
    
    def _get_access_token(self):
        """Get access token using gcloud command-line tool or environment."""
        if self.api_key:
            return self.api_key
            
        try:
            result = subprocess.run(
                [self.gcloud_path, 'auth', 'print-access-token'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logging.error(f"Error getting access token: {result.stderr}")
                return None
                
            token = result.stdout.strip()
            if not token:
                logging.error("No access token returned from gcloud")
                return None
                
            return token
        except Exception as e:
            logging.error(f"Error getting access token using gcloud: {e}")
            return None
    
    def _detect_audio_format(self, file_path):
        """
        Detect the audio format and sample rate of a file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            tuple: (encoding, sample_rate_hertz)
        """
        # Default values
        encoding = "FLAC"
        sample_rate_hertz = 16000
        
        # Check file extension
        file_ext = Path(file_path).suffix.lower()
        
        format_map = {
            '.flac': "FLAC",
            '.wav': "LINEAR16", 
            '.mp3': "MP3",
            '.ogg': "OGG_OPUS"
        }
        
        encoding = format_map.get(file_ext, "FLAC")
        
        return encoding, sample_rate_hertz
    
    def transcribe(self, audio_path):
        """
        Transcribe audio using Google Speech-to-Text v1.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription results
        """
        try:
            # Get access token for authorization
            access_token = self._get_access_token()
            if not access_token:
                return {
                    'text': '',
                    'error': 'Failed to get access token'
                }
            
            # Detect audio format and sample rate
            encoding, sample_rate_hertz = self._detect_audio_format(audio_path)
            
            # Read and encode audio file
            with open(audio_path, 'rb') as audio_file:
                audio_content = audio_file.read()
            
            audio_content_base64 = base64.b64encode(audio_content).decode('utf-8')
            
            # Prepare request
            request_body = {
                "config": {
                    "encoding": encoding,
                    "sampleRateHertz": sample_rate_hertz,
                    "languageCode": self.language_code,
                    "enableWordTimeOffsets": True,
                    "enableAutomaticPunctuation": True,
                    "model": "default",
                },
                "audio": {
                    "content": audio_content_base64
                }
            }
            
            # Set up headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
            
            if self.project_id:
                headers["x-goog-user-project"] = self.project_id
            
            # Make API request
            response = requests.post(
                "https://speech.googleapis.com/v2/speech:recognize",
                headers=headers,
                json=request_body
            )
            
            if response.status_code != 200:
                logging.error(f"Google API error ({response.status_code}): {response.text}")
                return {
                    'text': '',
                    'error': response.text
                }
            
            response_json = response.json()
            
            # Extract transcript and confidence
            transcript = ""
            confidence = 0
            chunks = []
            
            if "results" in response_json:
                for result in response_json["results"]:
                    if "alternatives" in result and result["alternatives"]:
                        alt = result["alternatives"][0]
                        transcript += alt.get("transcript", "") + " "
                        
                        # Get confidence from first result
                        if confidence == 0 and "confidence" in alt:
                            confidence = alt["confidence"]
                        
                        # Extract word timings
                        if "words" in alt:
                            for word_info in alt["words"]:
                                chunks.append({
                                    'word': word_info.get("word", ""),
                                    'start_time': float(word_info.get("startTime", "0s").rstrip("s")),
                                    'end_time': float(word_info.get("endTime", "0s").rstrip("s")),
                                    'confidence': confidence,
                                    'punctuated_word': word_info.get("word", "")
                                })
            
            return {
                'text': transcript.strip(),
                'chunks': chunks,
                'confidence': confidence
            }
            
        except Exception as e:
            logging.error(f"Error transcribing with Google Speech v1: {e}")
            return {
                'text': '',
                'error': str(e)
            }