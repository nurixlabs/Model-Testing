import os
import torch
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
        self.model_id = config.get('model_id', 'openai/whisper-large-v3-turbo')
        
        # Determine device
        if config.get('device'):
            self.device = config.get('device')
        else:
            # Auto-detect device
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.batch_size = config.get('batch_size', 1)
        self.language = config.get('language', 'en')
    
    def load(self):
        """Load the Whisper model from HuggingFace."""
        print(f"Loading Whisper model '{self.model_id}' on {self.device}...")
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            use_safetensors=True
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        generate_kwargs = {"language": self.language}
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
            return_timestamps="word"
        )
        
        print("Whisper model loaded successfully.")
    
    def transcribe(self, audio_path):
        """
        Transcribe the audio file using Whisper.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Dictionary containing:
                - text (str): The transcribed text
                - chunks (list): Word-level information with timing
        """
        try:
            result = self.pipe(audio_path)
            
            # Extract transcript text
            transcript = result.get('text', '')
            
            # Format word timestamps if they exist
            chunks = []
            if 'chunks' in result:
                for chunk in result['chunks']:
                    chunks.append({
                        'word': chunk.get('text', ''),
                        'start_time': chunk.get('timestamp', [0, 0])[0],
                        'end_time': chunk.get('timestamp', [0, 0])[1]
                    })
            
            return {
                'text': transcript,
                'chunks': chunks
            }
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return {'text': '', 'error': str(e)}