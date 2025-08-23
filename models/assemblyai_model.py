"""
AssemblyAI Speech-to-Text Model Implementation
"""
import os
import logging
from models.base_model import BaseModel
import assemblyai as aai


class AssemblyAIModel(BaseModel):
    """AssemblyAI speech-to-text API implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "assemblyai"
        self.client = None
        self.api_key = config.get('api_key', os.environ.get('ASSEMBLYAI_API_KEY'))
        self.language = config.get('language', 'en')
        self.punctuate = config.get('punctuate', True)
        self.format_text = config.get('format_text', True)
        self.speaker_labels = config.get('speaker_labels', False)
        self.auto_highlights = config.get('auto_highlights', False)
    
    def load(self):
        """Initialize the AssemblyAI client."""
        if not self.api_key:
            raise ValueError("AssemblyAI API key is required. Set it in config or ASSEMBLYAI_API_KEY environment variable.")
        
        logging.info("Initializing AssemblyAI client")
        logging.info(f"Language: {self.language}, Punctuate: {self.punctuate}")
        
        aai.settings.api_key = self.api_key
        self.client = aai.Transcriber()
        logging.info("AssemblyAI client initialized successfully")
    
    def transcribe(self, audio_path):
        """
        Transcribe audio using AssemblyAI.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription results
        """
        try:
            # For Hinglish, use Hindi as AssemblyAI doesn't support code-switching
            lang_code = 'hi' if self.language == 'hinglish' else self.language
            
            # Configure transcription options
            config = aai.TranscriptionConfig(
                punctuate=self.punctuate,
                format_text=self.format_text,
                speaker_labels=self.speaker_labels,
                auto_highlights=self.auto_highlights,
                language_code=lang_code if lang_code != 'en' else None
            )
            
            # Submit transcription and wait for completion
            transcript = self.client.transcribe(audio_path, config=config)
            
            # Check for errors
            if transcript.status == aai.TranscriptStatus.error:
                logging.error(f"AssemblyAI transcription failed: {transcript.error}")
                return {
                    'text': '',
                    'error': f'Transcription failed: {transcript.error}'
                }
            
            # Extract transcript text
            transcript_text = transcript.text or ''
            
            # Extract word timings and convert to standardized format
            chunks = []
            if hasattr(transcript, 'words') and transcript.words:
                for word in transcript.words:
                    chunks.append({
                        'word': word.text or '',
                        'start_time': (word.start or 0) / 1000.0,  # Convert ms to seconds
                        'end_time': (word.end or 0) / 1000.0,  # Convert ms to seconds
                        'confidence': word.confidence or 0,
                        'punctuated_word': word.text or ''
                    })
            
            # Get confidence score
            confidence = transcript.confidence or 0
            
            return {
                'text': transcript_text,
                'chunks': chunks,
                'confidence': confidence
            }
            
        except Exception as e:
            logging.error(f"Error transcribing with AssemblyAI: {e}")
            return {
                'text': '',
                'error': str(e)
            }