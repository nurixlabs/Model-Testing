#!/usr/bin/env python3
"""
Indic Conformer 600M Model for ASR Pipeline
Following BaseModel abstract class pattern
"""

import time
import torch
import torchaudio
from transformers import AutoModel
from models.base_model import BaseModel 

class IndicConformerModel(BaseModel):
    """Indic Conformer 600M multilingual ASR model"""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "indic-conformer"
        self.model = None
        self.device = None
        self.language = config.get('language', 'mr')  # Default to Marathi
        self.decoding_method = config.get('decoding_method', 'rnnt')
        self.target_sample_rate = config.get('target_sample_rate', 16000)
        self.model_name = config.get('model_name', 'ai4bharat/indic-conformer-600m-multilingual')
        
    def load(self):
        """Load the Indic Conformer model"""
        print(f"Loading Indic Conformer 600M model...")
        
        try:
            # Setup device
            device_config = self.config.get('device', 'auto')
            if device_config == 'auto':
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device_config)
            
            print(f"Using device: {self.device}")
            
            # Load model
            start_time = time.time()
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
            
            load_time = time.time() - start_time
            print(f"✅ Indic Conformer model loaded in {load_time:.2f} seconds")
            print(f"📊 Language: {self.language}")
            print(f"🔧 Decoding method: {self.decoding_method}")
            
        except Exception as e:
            print(f"❌ Failed to load Indic Conformer model: {e}")
            raise e
    
    def _preprocess_audio(self, audio_path):
        """Preprocess audio file"""
        try:
            # Load audio
            wav, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, 
                    new_freq=self.target_sample_rate
                )
                wav = resampler(wav)
            
            # Move to device
            wav = wav.to(self.device)
            
            return wav
            
        except Exception as e:
            print(f"❌ Audio preprocessing failed: {e}")
            return None
    
    def transcribe(self, audio_path):
        """
        Transcribe the audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Dictionary containing:
                - text (str): The transcribed text
                - chunks (list, optional): Word-level information with timing
                - confidence (float, optional): Confidence score
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Preprocess audio
            wav_tensor = self._preprocess_audio(audio_path)
            if wav_tensor is None:
                return {
                    'text': '',
                    'error': 'Failed to preprocess audio',
                    'confidence': 0.0
                }
            
            # Measure latency
            start_time = time.time()
            
            # Perform transcription
            with torch.no_grad():
                transcription = self.model(wav_tensor, self.language, self.decoding_method)
            
            latency = time.time() - start_time
            
            # Clean transcription
            transcription = transcription.strip() if transcription else ''
            
            # Return in the format expected by the pipeline
            result = {
                'text': transcription,  # Main transcription result
                'confidence': 1.0,      # Placeholder confidence score
                'latency': latency,
                'language': self.language,
                'decoding_method': self.decoding_method,
                'model_name': self.name
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Transcription failed for {audio_path}: {e}")
            return {
                'text': '',
                'error': str(e),
                'confidence': 0.0,
                'latency': 0.0
            }
    
    def get_model_info(self):
        """Return model information"""
        return {
            'name': self.name,
            'language': self.language,
            'decoding_method': self.decoding_method,
            'device': str(self.device) if self.device else 'unknown',
            'model_name': self.model_name
        }