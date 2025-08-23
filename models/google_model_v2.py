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
        self.language = config.get('language', 'english')
        
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
    
    def _split_and_transcribe(self, wav_path):
        """
        Split long audio into chunks and transcribe each chunk.
        
        Args:
            wav_path: Path to WAV audio file
            
        Returns:
            dict: Combined transcription results
        """
        try:
            # Load audio
            audio = AudioSegment.from_wav(wav_path)
            duration_ms = len(audio)
            
            # Split into 59-second chunks (59000ms)
            chunk_length_ms = 59000
            chunks_data = []
            
            for i in range(0, duration_ms, chunk_length_ms):
                # Extract chunk
                chunk = audio[i:i + chunk_length_ms]
                
                # Save chunk to temporary file
                chunk_fd, chunk_path = tempfile.mkstemp(suffix='.wav')
                os.close(chunk_fd)
                
                try:
                    chunk.export(chunk_path, format='wav')
                    
                    # Transcribe chunk
                    with open(chunk_path, "rb") as audio_file:
                        content = audio_file.read()
                    
                    # Set up recognition config
                    # For Hinglish, use both en-IN and hi-IN
                    if self.language == 'hinglish':
                        config = speech_v2.RecognitionConfig(
                            auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
                            language_codes=['en-IN', 'hi-IN'],  # Support both languages for code-switching
                            model=self.model,
                            features=speech_v2.RecognitionFeatures(
                                enable_automatic_punctuation=self.enable_punctuation,
                            ),
                        )
                    else:
                        config = speech_v2.RecognitionConfig(
                            auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
                            language_codes=self.language_codes,
                            model=self.model,
                            features=speech_v2.RecognitionFeatures(
                                enable_automatic_punctuation=self.enable_punctuation,
                            ),
                        )
                    
                    # Create request
                    request = speech_v2.RecognizeRequest(
                        config=config,
                        content=content,
                        recognizer=self.recognizer,
                    )
                    
                    # Get transcription
                    response = self.client.recognize(request=request)
                    
                    # Store chunk data with time offset
                    chunks_data.append({
                        'response': response,
                        'offset_seconds': i / 1000.0
                    })
                    
                finally:
                    # Clean up chunk file
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
            
            # Combine all chunk results
            full_transcript = ""
            all_chunks = []
            total_confidence = 0
            confidence_count = 0
            
            for chunk_data in chunks_data:
                response = chunk_data['response']
                offset = chunk_data['offset_seconds']
                
                for result in response.results:
                    if result.alternatives:
                        alt = result.alternatives[0]
                        full_transcript += alt.transcript + " "
                        
                        # Extract word-level information with adjusted timing
                        if hasattr(alt, 'words') and alt.words:
                            for word in alt.words:
                                all_chunks.append({
                                    'word': getattr(word, 'word', ''),
                                    'start_time': (word.start_offset.total_seconds() if hasattr(word, 'start_offset') else 0) + offset,
                                    'end_time': (word.end_offset.total_seconds() if hasattr(word, 'end_offset') else 0) + offset,
                                    'confidence': getattr(word, 'confidence', 0),
                                    'punctuated_word': getattr(word, 'word', '')
                                })
                        
                        # Accumulate confidence
                        if hasattr(alt, 'confidence'):
                            total_confidence += alt.confidence
                            confidence_count += 1
            
            full_transcript = full_transcript.strip()
            avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
            
            logging.info(f"Successfully transcribed {len(chunks_data)} chunks")
            
            return {
                'text': full_transcript,
                'chunks': all_chunks,
                'confidence': avg_confidence
            }
            
        except Exception as e:
            logging.error(f"Error in split_and_transcribe: {e}")
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
            
            # Check audio duration
            audio = AudioSegment.from_wav(wav_path)
            duration_seconds = len(audio) / 1000.0
            
            logging.info(f"Audio duration: {duration_seconds:.1f} seconds")
            
            # If audio is longer than 60 seconds, split and transcribe
            if duration_seconds > 60:
                logging.info(f"Audio exceeds 60 seconds, splitting into chunks...")
                return self._split_and_transcribe(wav_path)
            
            # For audio <= 60 seconds, use original single-pass method
            # Set up recognition config
            # For Hinglish, use both en-IN and hi-IN
            if self.language == 'hinglish':
                config = speech_v2.RecognitionConfig(
                    auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
                    language_codes=['en-IN', 'hi-IN'],  # Support both languages for code-switching
                    model=self.model,
                    features=speech_v2.RecognitionFeatures(
                        enable_automatic_punctuation=self.enable_punctuation,
                    ),
                )
            else:
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