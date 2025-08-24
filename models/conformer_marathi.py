"""
Conformer Marathi Model Implementation
"""

import os
import logging
import torch
from models.base_model import BaseModel
import re

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    nemo_asr = None


class ConformerMarathiModel(BaseModel):
    """Conformer Marathi speech-to-text model using NeMo Toolkit."""

    def __init__(self, config):
        super().__init__(config)
        self.name = "conformer_marathi"
        self.model = None
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_id = config.get(
            "model_id", "chintan-nurix/indicconformer_stt_multi_hybrid_rnnt_600m"
        )
        self.decoding_method = config.get("decoding_method", "ctc")
        self.language_code = config.get("language_code", "mr")

    def load(self):
        """Load the NeMo ASR model."""
        if not NEMO_AVAILABLE:
            raise ImportError("NeMo toolkit is not available")
            
        try:
            # Method 1: Try loading from local .nemo file
            if self.model_id.endswith(".nemo"):
                self.model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=self.model_id
                )
                logging.info("Model loaded from local .nemo file")
            else:
                # Method 2: Try from_pretrained for HuggingFace models
                self.model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_id
                )
                logging.info("Model loaded from HuggingFace")

            self.prepare_model()
        except Exception as e:
            logging.error(f"Failed to load NeMo model: {e}")
            raise

    def prepare_model(self):
        """Prepare the model for use with specific decoding method."""
        if hasattr(self.model, "cur_decoder"):
            if self.decoding_method == "ctc":
                self.model.cur_decoder = "ctc"
            elif self.decoding_method == "rnnt":
                self.model.cur_decoder = "rnnt"
            else:
                raise ValueError(f"Invalid decoding method: {self.decoding_method}")
        else:
            logging.warning("Model does not support decoding method")

    def transcribe(self, audio_path):
        """Transcribe audio using NeMo model."""
        if not self.model:
            self.load()
        if not os.path.exists(audio_path):
            return {"text": "", "error": f"Audio file not found: {audio_path}"}

        try:
            output = self.model.transcribe(
                [audio_path],
                batch_size=1,
                logprobs=False,
                language_id=self.language_code,
            )[0]
            if not output:
                return {"text": "", "error": "No transcription output received"}

            transcript_text = output[0]
            return {"text": transcript_text, "chunks": []}

        except Exception as e:
            logging.error(f"Error transcribing with NeMo model: {e}")
            return {"text": "", "error": str(e)}
