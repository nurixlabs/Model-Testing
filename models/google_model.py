import os
import base64
import requests
import subprocess
from models.base_model import BaseModel
from pathlib import Path

class GoogleModel(BaseModel):
    """Google Speech-to-Text API implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "google"
        self.api_key = os.environ.get('GOOGLE_API_KEY', config.get('api_key'))
        self.language_code = config.get('language_code', 'en-US')
    
    def load(self):
        """Initialize Google Speech-to-Text setup."""
        # Verify API key or Google Cloud authentication
        if not self.api_key and not self._check_gcloud_auth():
            print("WARNING: No Google API key provided and gcloud authentication not set up.")
            print("You will need to authenticate with Google Cloud before transcription.")
        else:
            print("Google Speech-to-Text initialized successfully.")
    
    def _check_gcloud_auth(self):
        """Check if gcloud authentication is set up."""
        try:
            result = subprocess.run(
                ['gcloud', 'auth', 'print-access-token'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return bool(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _get_access_token(self):
        """Get access token using gcloud command-line tool or environment."""
        if self.api_key:
            return self.api_key
            
        try:
            result = subprocess.run(
                ['gcloud', 'auth', 'print-access-token'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error getting access token using gcloud: {e}")
            print("Please set the GOOGLE_API_KEY environment variable or configure gcloud.")
            return None
    
    def _detect_audio_format(self, file_path):
        """
        Detect the audio format and sample rate of a file.
        
        Parameters:
            file_path (str): Path to the audio file.
            
        Returns:
            tuple: (encoding, sample_rate_hertz)
        """
        # Default values
        encoding = "FLAC"
        sample_rate_hertz = 16000
        
        # Check file extension
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.flac':
            encoding = "FLAC"
        elif file_ext == '.wav':
            encoding = "LINEAR16"
        elif file_ext == '.mp3':
            encoding = "MP3"
        elif file_ext == '.ogg':
            encoding = "OGG_OPUS"
        
        # For a more accurate approach, you'd want to use a library like pydub or ffprobe
        # to detect the sample rate
        
        return encoding, sample_rate_hertz
    
    def transcribe(self, audio_path):
        """
        Transcribe the audio file using Google Speech-to-Text.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Dictionary containing:
                - text (str): The transcribed text
                - confidence (float): Confidence score
        """
        try:
            # Get access token for authorization
            access_token = self._get_access_token()
            if not access_token:
                return {"error": "Failed to get access token", "text": ""}
            
            # Detect audio format and sample rate
            encoding, sample_rate_hertz = self._detect_audio_format(audio_path)
            
            # Read audio file as binary and encode as base64
            with open(audio_path, 'rb') as audio_file:
                audio_content = audio_file.read()
            
            audio_content_base64 = base64.b64encode(audio_content).decode('utf-8')
            
            # Prepare the request body
            request_body = {
                "config": {
                    "encoding": encoding,
                    "sampleRateHertz": sample_rate_hertz,
                    "languageCode": self.language_code,
                    "enableWordTimeOffsets": True,
                    "enableAutomaticPunctuation": True
                },
                "audio": {
                    "content": audio_content_base64
                }
            }
            
            # Make the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
            
            response = requests.post(
                "https://speech.googleapis.com/v1/speech:recognize",
                headers=headers,
                json=request_body
            )
            
            if response.status_code != 200:
                print(f"Error response ({response.status_code}): {response.text}")
                return {"error": response.text, "text": ""}
            
            response_json = response.json()
            
            # Extract transcript text
            transcript = ""
            confidence = 0
            
            if "results" in response_json:
                for result in response_json["results"]:
                    if "alternatives" in result and result["alternatives"]:
                        alt = result["alternatives"][0]
                        transcript += alt["transcript"] + " "
                        if "confidence" in alt:
                            # For overall confidence, we'll just use the confidence of the first result
                            # A more sophisticated approach would weight by length of each segment
                            if confidence == 0:
                                confidence = alt["confidence"]
            
            # Extract word timings
            words = []
            if "results" in response_json:
                for result in response_json["results"]:
                    if "alternatives" in result and result["alternatives"]:
                        alt = result["alternatives"][0]
                        if "words" in alt:
                            for word_info in alt["words"]:
                                word = {
                                    "word": word_info["word"],
                                    "start_time": float(word_info["startTime"].rstrip("s")),
                                    "end_time": float(word_info["endTime"].rstrip("s"))
                                }
                                words.append(word)
            
            return {
                "text": transcript.strip(),
                "confidence": confidence,
                "chunks": words,
                "raw_response": response_json
            }
            
        except Exception as e:
            print(f"Error transcribing with Google STT: {e}")
            return {"text": "", "error": str(e)}