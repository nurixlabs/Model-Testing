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
from typing import Optional, Tuple


class GoogleModel(BaseModel):
    """Google Speech-to-Text v1 API implementation."""

    def __init__(self, config):
        super().__init__(config)
        self.name = "google"
        self.api_key = config.get('api_key', os.environ.get('GOOGLE_API_KEY'))
        self.language_code = config.get('language_code', 'en-US')
        self.project_id = config.get('project_id', os.environ.get('GOOGLE_PROJECT_ID'))
        self.gcloud_path = config.get('gcloud_path', './google-cloud-sdk/bin/gcloud')
        self.default_sample_rate_hz = int(config.get('default_sample_rate_hz', 16000))
        # NEW: only send x-goog-user-project if you explicitly opt-in
        self.use_quota_project = bool(config.get('use_quota_project', False))

    def load(self):
        """Initialize Google Speech-to-Text v1."""
        logging.info("Initializing Google Speech-to-Text v1")
        logging.info(f"Language: {self.language_code}, Project ID: {self.project_id}")
        logging.info(f"gcloud path: {self.gcloud_path}")

        if not self.api_key and not self._check_gcloud_auth():
            logging.warning("No Google API key provided and gcloud authentication not set up")
            logging.warning("You will need to authenticate with Google Cloud before transcription")
        else:
            logging.info("Google Speech-to-Text initialized successfully")

        if self.use_quota_project and not self.project_id:
            logging.warning("use_quota_project=True but no project_id set; the header won't be added.")

    def _check_gcloud_auth(self) -> bool:
        """Check if gcloud authentication is set up."""
        try:
            result = subprocess.run(
                [self.gcloud_path, "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
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

    def _get_access_token(self) -> Optional[str]:
        """
        Get OAuth access token (used when no API key is provided).
        For v1 you can use either API key or OAuth.
        """
        if self.api_key:
            return None  # using API key, so no bearer token
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

    def _detect_audio_format(self, file_path: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Detect the audio format/sample rate for RecognitionConfig.
        For FLAC/WAV we typically omit encoding/rate (v1 can infer from headers).
        """
        ext = Path(file_path).suffix.lower()
        if ext in ('.flac', '.wav'):
            return None, None  # let v1 infer
        if ext == '.mp3':
            return "MP3", self.default_sample_rate_hz
        if ext in ('.ogg', '.opus'):
            return "OGG_OPUS", self.default_sample_rate_hz
        if ext in ('.pcm', '.raw'):
            return "LINEAR16", self.default_sample_rate_hz
        return None, None

    @staticmethod
    def _parse_time_s(value: str) -> float:
        """Parse '123.45s' to float seconds."""
        try:
            if isinstance(value, str) and value.endswith('s'):
                return float(value[:-1]) if value[:-1] else 0.0
            return float(value)
        except Exception:
            return 0.0

    def transcribe(self, audio_path: str):
        """
        Transcribe audio using Google Speech-to-Text v1.

        Returns:
            dict: { 'text': str, 'chunks': list[dict], 'confidence': float, 'error'?: str }
        """
        try:
            # Auth
            access_token = self._get_access_token()

            # Read audio
            with open(audio_path, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')

            # Detect encoding/sample rate
            encoding, sample_rate_hz = self._detect_audio_format(audio_path)

            # Build v1 request
            config = {
                "languageCode": self.language_code,
                "enableWordTimeOffsets": True,
                "enableAutomaticPunctuation": True,
                "model": "default",
            }
            if encoding:
                config["encoding"] = encoding
            if sample_rate_hz:
                config["sampleRateHertz"] = sample_rate_hz

            request_body = {"config": config, "audio": {"content": audio_b64}}

            url = "https://speech.googleapis.com/v1/speech:recognize"
            headers = {"Content-Type": "application/json"}
            params = None

            if self.api_key:
                # API key auth
                params = {"key": self.api_key}
            else:
                # OAuth bearer
                if not access_token:
                    return {'text': '', 'error': 'Failed to get access token'}
                headers["Authorization"] = f"Bearer {access_token}"

            # IMPORTANT: Only add x-goog-user-project if explicitly requested
            if self.use_quota_project and self.project_id:
                headers["x-goog-user-project"] = self.project_id

            # Call v1
            response = requests.post(url, headers=headers, params=params, json=request_body)

            if response.status_code != 200:
                logging.error(f"Google API error ({response.status_code}): {response.text}")
                return {'text': '', 'error': response.text}

            data = response.json()

            transcript_parts = []
            confidence = 0.0
            chunks = []

            for result in data.get("results", []):
                alts = result.get("alternatives", [])
                if not alts:
                    continue
                top = alts[0]
                if top.get("transcript"):
                    transcript_parts.append(top["transcript"])
                if not confidence and "confidence" in top:
                    confidence = top.get("confidence") or 0.0

                for w in top.get("words", []):
                    chunks.append({
                        "word": w.get("word", ""),
                        "start_time": self._parse_time_s(w.get("startTime", "0s")),
                        "end_time": self._parse_time_s(w.get("endTime", "0s")),
                        "confidence": w.get("confidence", confidence) or 0.0,
                        "punctuated_word": w.get("word", "")
                    })

            return {
                'text': " ".join(transcript_parts).strip(),
                'chunks': chunks,
                'confidence': confidence
            }

        except Exception as e:
            logging.error(f"Error transcribing with Google Speech v1: {e}")
            return {'text': '', 'error': str(e)}