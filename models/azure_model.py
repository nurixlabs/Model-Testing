"""
Azure Speech-to-Text Model Implementation
"""
import os
import tempfile
import logging
import wave
from models.base_model import BaseModel
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment


class AzureModel(BaseModel):
    """Azure Speech-to-Text API implementation with robust format handling."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "azure"
        self.speech_config = None
        self.subscription_key = config.get('subscription_key', os.environ.get('AZURE_SPEECH_KEY'))
        self.region = config.get('region', 'eastus')
        self.language = config.get('language', 'en-US')
        self.enable_word_timing = config.get('enable_word_timing', True)
        self.enable_punctuation = config.get('enable_punctuation', True)
    
    def load(self):
        """Initialize the Azure Speech client configuration."""
        if not self.subscription_key:
            raise ValueError("Azure subscription key is required. Set it in config or AZURE_SPEECH_KEY environment variable.")
        
        logging.info("Initializing Azure Speech client")
        logging.info(f"Region: {self.region}, Language: {self.language}")
        
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.subscription_key,
            region=self.region
        )
        
        self.speech_config.speech_recognition_language = self.language
        
        if self.enable_word_timing:
            self.speech_config.request_word_level_timestamps()
        
        if self.enable_punctuation:
            self.speech_config.enable_dictation()
        
        logging.info("Azure Speech client initialized successfully")
    
    def _validate_wav_file(self, wav_path):
        """Validate WAV file format for Azure Speech."""
        try:
            with wave.open(wav_path, 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                
                # Azure Speech requirements
                if channels > 2:
                    return False, "Too many channels (max 2)"
                if sample_width != 2:  # 16-bit
                    return False, "Not 16-bit audio"
                if framerate not in [8000, 16000, 24000, 48000]:
                    return False, f"Unsupported sample rate: {framerate}"
                
                return True, "Valid"
        except Exception as e:
            return False, f"Invalid WAV file: {e}"
    
    def _convert_to_azure_format(self, input_path):
        """Convert audio to Azure-compatible format."""
        try:
            ext = os.path.splitext(input_path)[1].lower()
            
            if ext == '.wav':
                is_valid, msg = self._validate_wav_file(input_path)
                if is_valid:
                    return input_path
            
            # Load audio with appropriate format
            if ext == '.mp3':
                audio = AudioSegment.from_mp3(input_path)
            elif ext == '.ogg':
                audio = AudioSegment.from_ogg(input_path)
            elif ext == '.mp4':
                audio = AudioSegment.from_file(input_path, format='mp4')
            elif ext == '.wav':
                audio = AudioSegment.from_wav(input_path)
            else:
                audio = AudioSegment.from_file(input_path)
            
            # Convert to Azure-compatible format: 16-bit PCM, 16 kHz, mono
            audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)
            
            # Export to temporary WAV file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            
            audio.export(temp_path, format='wav', parameters=[
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1'
            ])
            
            # Validate conversion
            is_valid, msg = self._validate_wav_file(temp_path)
            if not is_valid:
                raise ValueError(f"Conversion failed: {msg}")
            
            return temp_path
            
        except Exception as e:
            logging.error(f"Error converting audio: {e}")
            raise
    
    def transcribe(self, audio_path):
        """
        Transcribe audio using Azure Speech-to-Text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription results
        """
        temp_path = None
        try:
            # Convert audio to Azure-compatible format
            temp_path = self._convert_to_azure_format(audio_path)
            
            # Create audio configuration and recognizer
            audio_config = speechsdk.AudioConfig(filename=temp_path)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Perform recognition
            result = speech_recognizer.recognize_once()
            
            # Process results
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                detailed_results = result.best()
                chunks = []
                confidence = 0
                
                if detailed_results and len(detailed_results) > 0:
                    best_result = detailed_results[0]
                    confidence = best_result.confidence
                    
                    # Extract word-level timing
                    if hasattr(best_result, 'words') and best_result.words:
                        for word in best_result.words:
                            # Convert ticks to seconds (1 tick = 100 nanoseconds)
                            start_time = word.offset / 10000000.0
                            duration = word.duration / 10000000.0
                            end_time = start_time + duration
                            
                            chunks.append({
                                'word': word.word,
                                'start_time': start_time,
                                'end_time': end_time,
                                'confidence': confidence,
                                'punctuated_word': word.word
                            })
                
                return {
                    'text': result.text,
                    'chunks': chunks,
                    'confidence': confidence
                }
            
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logging.error(f"Azure: No speech recognized - {result.no_match_details}")
                return {
                    'text': '',
                    'error': f'No speech could be recognized: {result.no_match_details}'
                }
            
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                error_msg = f'Speech recognition canceled: {cancellation_details.reason}'
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    error_msg += f' Error details: {cancellation_details.error_details}'
                
                logging.error(f"Azure transcription canceled: {error_msg}")
                return {
                    'text': '',
                    'error': error_msg
                }
            
            else:
                logging.error(f"Azure: Unexpected result reason - {result.reason}")
                return {
                    'text': '',
                    'error': f'Unexpected result reason: {result.reason}'
                }
                
        except Exception as e:
            logging.error(f"Error transcribing with Azure: {e}")
            return {
                'text': '',
                'error': str(e)
            }
        finally:
            # Clean up temporary file
            if temp_path and temp_path != audio_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass