"""
Google Speech-to-Text Chirp 2 Model Implementation
"""
import os
import sys
import tempfile
import logging
from models.base_model import BaseModel
from google.cloud import speech_v2
from google.api_core.client_options import ClientOptions
from pydub import AudioSegment

# Add parent directory to path to import google_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from google_utils.google_credentials import get_google_credentials, setup_google_environment



class GoogleChirp2Model(BaseModel):
    """Google Speech-to-Text Chirp 2 API implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "google_chirp2"
        self.client = None
        self.project_id = config.get('project_id', 'train-453515')
        self.location = config.get('location', 'us-central1')
        self.api_endpoint = f"{self.location}-speech.googleapis.com"
        self.language_codes = config.get('language_codes', ['en-IN'])
        self.model = config.get('model', 'chirp_2')
        self.enable_punctuation = config.get('enable_punctuation', True)
        
        # Set up Google credentials from environment
        setup_google_environment()
        self.credentials = get_google_credentials()
    
    def load(self):
        """Initialize the Google Speech-to-Text client."""
        logging.info("Initializing Google Chirp 2 client")
        logging.info(f"Project: {self.project_id}, Location: {self.location}")
        logging.info(f"Model: {self.model}, Languages: {self.language_codes}")
        
        try:
            # Create client with credentials if available
            if self.credentials:
                self.client = speech_v2.SpeechClient(
                    credentials=self.credentials,
                    client_options=ClientOptions(api_endpoint=self.api_endpoint)
                )
            else:
                # Fall back to default credentials
                self.client = speech_v2.SpeechClient(
                    client_options=ClientOptions(api_endpoint=self.api_endpoint)
                )
            self.recognizer = self.client.recognizer_path(
                self.project_id, self.location, "_"
            )
            logging.info("Google Chirp 2 client initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Google Chirp 2 client: {e}")
            raise
    
    def _convert_to_wav(self, input_path):
        """Convert audio to WAV format if needed."""
        ext = os.path.splitext(input_path)[1].lower()
        
        if ext == '.wav':
            return input_path
        
        # Convert to WAV
        try:
            if ext == '.ogg':
                audio = AudioSegment.from_ogg(input_path)
            elif ext == '.mp4':
                audio = AudioSegment.from_file(input_path, format='mp4')
            elif ext == '.mp3':
                audio = AudioSegment.from_mp3(input_path)
            else:
                audio = AudioSegment.from_file(input_path)
            
            # Create temporary WAV file
            wav_fd, wav_path = tempfile.mkstemp(suffix='.wav')
            os.close(wav_fd)
            
            # Convert to 16-bit PCM WAV at 16kHz, mono
            audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)
            audio.export(wav_path, format='wav')
            
            return wav_path
            
        except Exception as e:
            logging.error(f"Error converting audio to WAV: {e}")
            raise
    
    def transcribe(self, audio_path):
        """
        Transcribe audio using Google Chirp 2.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription results
        """
        wav_path = None
        try:
            # Convert to WAV if needed
            wav_path = self._convert_to_wav(audio_path)
            
            # Set up recognition config
            config = speech_v2.RecognitionConfig(
                auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
                language_codes=self.language_codes,
                model=self.model,
                features=speech_v2.RecognitionFeatures(
                    enable_automatic_punctuation=self.enable_punctuation,
                ),
            )
            
            # Read audio file
            with open(wav_path, "rb") as audio_file:
                content = audio_file.read()
            
            # Create request
            request = speech_v2.RecognizeRequest(
                config=config,
                content=content,
                recognizer=self.recognizer,
            )
            
            # Get transcription
            response = self.client.recognize(request=request)
            
            # Process results
            if not response.results:
                return {
                    'text': '',
                    'error': 'No speech detected in audio'
                }
            
            transcript = ""
            chunks = []
            total_confidence = 0
            confidence_count = 0
            
            for result in response.results:
                if result.alternatives:
                    alt = result.alternatives[0]
                    transcript += alt.transcript + " "
                    
                    # Extract word-level information
                    if hasattr(alt, 'words') and alt.words:
                        for word in alt.words:
                            chunks.append({
                                'word': getattr(word, 'word', ''),
                                'start_time': word.start_offset.total_seconds() if hasattr(word, 'start_offset') else 0,
                                'end_time': word.end_offset.total_seconds() if hasattr(word, 'end_offset') else 0,
                                'confidence': getattr(word, 'confidence', 0),
                                'punctuated_word': getattr(word, 'word', '')
                            })
                    
                    # Accumulate confidence
                    if hasattr(alt, 'confidence'):
                        total_confidence += alt.confidence
                        confidence_count += 1
            
            transcript = transcript.strip()
            avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
            
            return {
                'text': transcript,
                'chunks': chunks,
                'confidence': avg_confidence
            }
                
        except Exception as e:
            logging.error(f"Error transcribing with Google Chirp 2: {e}")
            return {
                'text': '',
                'error': str(e)
            }
        finally:
            # Clean up temporary WAV file if created
            if wav_path and wav_path != audio_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except:
                    pass