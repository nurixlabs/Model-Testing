"""
Cartesia Ink Whisper Model Implementation
High-quality transcription using Cartesia's Whisper API (Batch STT)
"""

import os
import time
import logging
import requests
import mimetypes

# Optional MIME detectors
_HAS_MAGIC = False
_HAS_FILETYPE = False
try:
    import magic  # python-magic (libmagic)
    _HAS_MAGIC = True
except Exception:
    pass
try:
    import filetype  # h2non/filetype.py
    _HAS_FILETYPE = True
except Exception:
    pass

from models.base_model import BaseModel

# Seed common audio types (some OS images miss these)
mimetypes.add_type("audio/mp4", ".m4a")
mimetypes.add_type("audio/ogg", ".ogg")
mimetypes.add_type("audio/ogg", ".oga")
mimetypes.add_type("audio/opus", ".opus")
mimetypes.add_type("audio/aac", ".aac")
mimetypes.add_type("audio/wav", ".wav")
mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("audio/mpeg", ".mp3")
mimetypes.add_type("audio/webm", ".webm")
mimetypes.add_type("audio/amr", ".amr")
mimetypes.add_type("audio/aiff", ".aiff")
mimetypes.add_type("audio/aiff", ".aif")
mimetypes.add_type("audio/x-ms-wma", ".wma")
mimetypes.add_type("audio/x-matroska", ".mka")
mimetypes.add_type("audio/x-caf", ".caf")


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return key[:2] + "…" + key[-2:]
    return key[:4] + "…" + key[-4:]


def _guess_mime_type(file_path: str) -> str:
    """
    Best-effort MIME detection order:
      1) libmagic (header-based)
      2) filetype.py (magic numbers)
      3) mimetypes (extension)
      4) manual fallback by extension
    """
    if _HAS_MAGIC:
        try:
            m = magic.Magic(mime=True)
            mime = m.from_file(file_path)
            if mime:
                return mime
        except Exception:
            pass

    if _HAS_FILETYPE:
        try:
            kind = filetype.guess(file_path)
            if kind and kind.mime:
                return kind.mime
        except Exception:
            pass

    mime, _enc = mimetypes.guess_type(file_path)
    if mime:
        return mime

    ext = os.path.splitext(file_path)[1].lower()
    manual = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".oga": "audio/ogg",
        ".opus": "audio/opus",
        ".webm": "audio/webm",
        ".amr": "audio/amr",
        ".aiff": "audio/aiff",
        ".aif": "audio/aiff",
        ".wma": "audio/x-ms-wma",
        ".mka": "audio/x-matroska",
        ".caf": "audio/x-caf",
    }
    return manual.get(ext, "application/octet-stream")


class CartesiaInkWhisperModel(BaseModel):
    """Cartesia Ink Whisper speech-to-text implementation (Batch STT)."""

    def __init__(self, config):
        super().__init__(config)
        self.name = "cartesia"
        # Env var takes precedence per your original code
        self.api_key = os.environ.get('CARTESIA_API_KEY', config.get('api_key'))
        self.model = config.get('model', 'ink-whisper')
        self.language = config.get('language', 'en')
        self.api_base_url = config.get('api_base_url', 'https://api.cartesia.ai')
        # Accept either "word" or ["word"]; will normalize to list later
        self.timestamp_granularities = config.get('timestamp_granularities[]', 'word')
        # Optional advanced params supported by API
        self.encoding = config.get('encoding')          # e.g., "pcm_s16le"
        self.sample_rate = config.get('sample_rate')    # e.g., 16000
        # Required API version header per docs
        self.api_version = config.get('api_version', '2025-04-16')

    def load(self):
        """Initialize Cartesia API."""
        if not self.api_key:
            raise ValueError(
                "Cartesia API key is required. "
                "Set it in config or CARTESIA_API_KEY environment variable."
            )
        logging.info("Cartesia Ink Whisper API initialized")
        logging.info(f"Model: {self.model}, Language: {self.language}")
        logging.info(f"Cartesia-Version: {self.api_version}")
        logging.info(f"API key: { _mask_key(self.api_key) }")

    def _build_headers(self):
        # Per docs: Bearer auth + Cartesia-Version
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Cartesia-Version": self.api_version,
        }

    def _normalize_timestamp_granularities(self):
        # Docs expect timestamp_granularities[] as a list; only "word" is supported today
        tg = self.timestamp_granularities
        if isinstance(tg, str):
            tg = [tg]
        # filter/normalize
        return [g for g in tg if g] or ["word"]

    def _post_stt(self, file_path: str):
        """
        Send multipart/form-data to /stt with part name 'file', plus form fields.
        Returns: response.json() or raises on HTTP error.
        """
        url = f"{self.api_base_url}/stt"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"Audio file is empty: {file_path}")

        filename = os.path.basename(file_path)
        mime = _guess_mime_type(file_path)
        logging.info(f"Uploading for transcription: {filename} (mime={mime})")

        fields = {
            "model": self.model,
            "language": self.language,
        }

        # timestamp_granularities[] must be sent as repeated form fields
        for g in self._normalize_timestamp_granularities():
            fields.setdefault("timestamp_granularities[]", [])
            fields["timestamp_granularities[]"].append(g)

        # Optional query parameters (encoding, sample_rate) per docs
        # These are query params in docs, but the sample also shows form usage.
        # We'll include them as query params to match the reference.
        params = {}
        if self.encoding:
            params["encoding"] = self.encoding
        if self.sample_rate:
            params["sample_rate"] = str(self.sample_rate)

        # Build multipart: part name MUST be 'file'
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, mime)}
            resp = requests.post(
                url,
                headers=self._build_headers(),
                params=params,          # encoding & sample_rate
                data=fields,            # model, language, timestamp_granularities[]
                files=files,
                timeout=120,
            )

        if resp.status_code != 200:
            # Log details to help debug auth/headers/multipart issues
            logging.error(f"Cartesia API error: {resp.status_code}")
            logging.error(f"Response: {resp.text}")
            try:
                redacted_headers = {
                    k: v for k, v in resp.request.headers.items()
                    if k.lower() not in ("authorization",)
                }
                logging.debug(f"Request headers: {redacted_headers}")
            except Exception:
                pass
            resp.raise_for_status()

        return resp.json()

    def transcribe(self, audio_path=None, audio_url=None):
        """
        Transcribe audio using Cartesia Ink Whisper (Batch STT).
        Preferred usage: provide `audio_path` to a local file.
        """
        try:
            if not audio_path and audio_url:
                # Batch STT expects a file upload (multipart 'file').
                # If a URL is provided, you could download it to a temp file and then upload.
                # Keeping behavior explicit: require audio_path for now to match API.
                logging.warning(
                    "audio_url provided but batch STT expects a file upload. "
                    "Please download the audio locally and pass audio_path."
                )
                return {'text': '', 'error': 'Provide audio_path (multipart upload required).'}

            if not audio_path:
                return {'text': '', 'error': 'audio_path is required for Cartesia batch STT.'}

            result = self._post_stt(audio_path)

            # Parse result shape per docs
            transcript = result.get("text", "") or ""
            chunks = []
            words = result.get("words") or []

            for w in words:
                # docs: words have start/end (seconds) and word text
                chunks.append({
                    "word": w.get("word", ""),
                    "start_time": w.get("start", 0),
                    "end_time": w.get("end", 0),
                    # confidence not specified in docs; default to None/0
                    "confidence": w.get("confidence", 0),
                })

            return {
                "text": transcript,
                "chunks": chunks,
                "model": self.model,
                "language": self.language,
            }

        except requests.HTTPError as http_err:
            return {"text": "", "error": f"HTTP error: {http_err}"}
        except Exception as e:
            logging.error(f"Error transcribing with Cartesia: {e}")
            return {"text": "", "error": str(e)}