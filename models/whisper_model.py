"""
Whisper Speech-to-Text Model Implementation
"""
import os
import logging
import torch
import gc
from models.base_model import BaseModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperModel(BaseModel):
    """Whisper model implementation using HuggingFace transformers."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "whisper"
        self.model = None
        self.processor = None
        self.pipe = None
        self.model_id = config.get('model_id', 'openai/whisper-large-v2')
        self.batch_size = config.get('batch_size', 1)
        self.language = config.get('language', 'en')
        
        # Device detection
        if config.get('device'):
            self.device = config.get('device')
        else:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
    
    def load(self):
        """Load the Whisper model from HuggingFace."""
        logging.info(f"Loading Whisper model: {self.model_id}")
        logging.info(f"Device: {self.device}, Language: {self.language}, Batch size: {self.batch_size}")
        
        try:
            # Clear cache before loading
            self._clear_cache()
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True
            )
            self.model.to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Only set language for pure Hindi/Marathi, use auto-detection for English/Hinglish
            generate_kwargs = {}
            if self.language and self.language not in ['en', 'english', 'English', 'hinglish', 'Hinglish']:
                # Map language codes appropriately (only for pure languages)
                lang_map = {
                    'hi': 'hindi',
                    'hi-IN': 'hindi', 
                    'hindi': 'hindi',
                    'mr': 'marathi',
                    'mr-IN': 'marathi',
                    'marathi': 'marathi'
                }
                mapped_lang = lang_map.get(self.language, self.language)
                generate_kwargs = {"language": mapped_lang}
                logging.info(f"Whisper using language: {mapped_lang}")
            else:
                # Use auto-detection for English and Hinglish
                if self.language in ['hinglish', 'Hinglish']:
                    logging.info("Whisper using auto-detection for Hinglish (code-switching)")
                else:
                    logging.info("Whisper using auto-detection for English")
            
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                chunk_length_s=30,
                batch_size=self.batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps=True  # Use True instead of "word"
            )
            
            logging.info("Whisper model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _clear_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def transcribe(self, audio_path):
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription results
        """
        try:
            # Use no_grad context to prevent gradient computation
            with torch.no_grad():
                result = self.pipe(audio_path)
            
            # Extract transcript text
            transcript = result.get('text', '')
            
            # Format word timestamps
            chunks = []
            if 'chunks' in result and result['chunks']:
                for chunk in result['chunks']:
                    # Handle different timestamp formats
                    timestamp = chunk.get('timestamp', None)
                    start_time = 0
                    end_time = 0
                    
                    if timestamp is not None:
                        if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                            # Format: [start, end]
                            start_time = float(timestamp[0]) if timestamp[0] is not None else 0
                            end_time = float(timestamp[1]) if timestamp[1] is not None else 0
                        elif isinstance(timestamp, dict):
                            # Format: {'start': ..., 'end': ...}
                            start_time = float(timestamp.get('start', 0)) if timestamp.get('start') is not None else 0
                            end_time = float(timestamp.get('end', 0)) if timestamp.get('end') is not None else 0
                    
                    chunks.append({
                        'word': chunk.get('text', '').strip(),
                        'start_time': start_time,
                        'end_time': end_time,
                        'confidence': 0,  # Whisper doesn't provide word-level confidence
                        'punctuated_word': chunk.get('text', '').strip()
                    })
            
            return {
                'text': transcript,
                'chunks': chunks,
                'confidence': 0  # Whisper doesn't provide overall confidence scores
            }
            
        except Exception as e:
            logging.error(f"Error transcribing with Whisper: {e}")
            return {
                'text': '',
                'error': str(e)
            }
        finally:
            # Clear cache after inference
            self._clear_cache()