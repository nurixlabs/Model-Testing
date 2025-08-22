"""
Gladia Speech-to-Text Model Implementation
- Proper multipart upload with detected MIME types
- v2 pre-recorded flow with polling by job ID
"""

import os
import time
import logging
import requests
import mimetypes

# Optional MIME detectors (use if available)
_HAS_MAGIC = False
_HAS_FILETYPE = False
try:
    import magic  # python-magic / libmagic
    _HAS_MAGIC = True
except Exception:
    pass

try:
    import filetype  # h2non/filetype.py
    _HAS_FILETYPE = True
except Exception:
    pass

from models.base_model import BaseModel


# Seed/augment mimetypes for common audio extensions that can be missing on some systems
mimetypes.add_type("audio/mp4", ".m4a")
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
    Best-effort MIME detection:
      1) libmagic (header-based)
      2) filetype.py (magic numbers)
      3) mimetypes (extension)
      4) manual fallback map
    """
    # 1) libmagic
    if _HAS_MAGIC:
        try:
            m = magic.Magic(mime=True)
            mime = m.from_file(file_path)
            if mime:
                return mime
        except Exception:
            pass

    # 2) filetype.py
    if _HAS_FILETYPE:
        try:
            kind = filetype.guess(file_path)
            if kind and kind.mime:
                return kind.mime
        except Exception:
            pass

    # 3) mimetypes
    mime, _enc = mimetypes.guess_type(file_path)
    if mime:
        return mime

    # 4) manual conservative fallback by extension
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


class GladiaModel(BaseModel):
    """Gladia API speech-to-text implementation."""

    def __init__(self, config):
        super().__init__(config)
        self.name = "gladia"
        self.api_key = config.get('api_key', os.environ.get('GLADIA_API_KEY'))
        self.max_retries = config.get('max_retries', 60)
        self.poll_interval = config.get('poll_interval', 5)
        self.upload_url = "https://api.gladia.io/v2/upload"
        self.transcribe_url = "https://api.gladia.io/v2/pre-recorded"

    def load(self):
        """Initialize the Gladia client."""
        if not self.api_key:
            raise ValueError("Gladia API key is required. Set it in config or GLADIA_API_KEY environment variable.")

        logging.info("Initializing Gladia API client")
        logging.info(f"API key configured: { _mask_key(self.api_key) }")
        logging.info("Gladia API client initialized successfully")

    def _upload_file(self, file_path: str):
        """Upload a local file to Gladia and get an audio URL."""
        headers = {"x-gladia-key": self.api_key}

        try:
            if not os.path.exists(file_path):
                logging.error(f"File does not exist: {file_path}")
                return None

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logging.error(f"File is empty: {file_path}")
                return None

            filename = os.path.basename(file_path)
            mime = _guess_mime_type(file_path)
            logging.info(f"Uploading file: {filename} ({file_size} bytes, mime={mime})")

            # IMPORTANT: use 3-tuple so Requests sets the per-part Content-Type
            with open(file_path, "rb") as f:
                files = {"audio": (filename, f, mime)}
                resp = requests.post(self.upload_url, headers=headers, files=files, timeout=60)

            if resp.status_code not in (200, 201):
                logging.error(f"Gladia file upload error: {resp.status_code}")
                logging.error(f"Response: {resp.text}")
                # Helpful when diagnosing multipart/form-data issues:
                try:
                    logging.debug(f"Request headers (sans key): "
                                  f"{ {k:v for k,v in resp.request.headers.items() if k.lower() != 'x-gladia-key'} }")
                except Exception:
                    pass
                return None

            data = resp.json()
            audio_url = data.get("audio_url")
            logging.info(f"Upload successful, audio_url: {audio_url}")
            return audio_url

        except Exception as e:
            logging.error(f"Error uploading file to Gladia: {e}")
            return None

    def _submit_transcription_job(self, audio_url: str):
        """Submit transcription job to Gladia API (v2)."""
        headers = {
            "x-gladia-key": self.api_key,
            "Content-Type": "application/json",  # v2 requires JSON for /v2/pre-recorded
        }
        payload = {"audio_url": audio_url}

        logging.info("Submitting transcription job to /v2/pre-recorded")
        resp = requests.post(self.transcribe_url, headers=headers, json=payload, timeout=30)

        if resp.status_code not in (200, 201):
            logging.error(f"Gladia job submission error: {resp.status_code}")
            logging.error(f"Response: {resp.text}")
            return None

        data = resp.json()
        job_id = data.get("id")
        if not job_id:
            logging.error(f"Gladia response missing job id: {data}")
            return None

        logging.info(f"Job submitted successfully: id={job_id}")
        return job_id

    def _get_job_status(self, job_id: str):
        """Get job status from Gladia API (v2, poll by ID)."""
        url = f"{self.transcribe_url}/{job_id}"
        headers = {"x-gladia-key": self.api_key}

        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            logging.error(f"Gladia status check error: {resp.status_code}")
            logging.error(f"Response: {resp.text}")
            return None

        result = resp.json()
        logging.debug(f"Status check response: {result}")
        return result

    def transcribe(self, audio_input):
        """
        Transcribe audio using Gladia API.

        Args:
            audio_input: local file path or a URL to an audio file

        Returns:
            dict: { 'text': str, 'chunks': list, 'confidence': float, 'error': Optional[str] }
        """
        try:
            if not self.api_key:
                return {'text': '', 'error': 'API key not configured'}

            # Determine if input is a local file path or URL (simple heuristic)
            audio_url = audio_input
            if os.path.isfile(audio_input):
                logging.info(f"Uploading local file: {audio_input}")
                audio_url = self._upload_file(audio_input)
                if not audio_url:
                    return {'text': '', 'error': 'Failed to upload audio file'}
                logging.info(f"File uploaded successfully, audio_url: {audio_url}")

            # Submit transcription job
            job_id = self._submit_transcription_job(audio_url)
            if not job_id:
                return {'text': '', 'error': 'Failed to submit transcription job'}

            logging.info(f"Submitted Gladia job {job_id}, waiting for completion...")

            job_status = None
            for retry in range(self.max_retries):
                logging.info(f"Polling for results... (attempt {retry + 1}/{self.max_retries})")
                job_status = self._get_job_status(job_id)
                if not job_status:
                    logging.warning("Failed to get job status, retrying...")
                    time.sleep(self.poll_interval)
                    continue

                status = str(job_status.get('status', '')).lower()
                logging.info(f"Job status: {status}")

                if status in ['done', 'succeeded', 'completed']:
                    logging.info("Transcription completed!")
                    break
                elif status in ['failed', 'error']:
                    error_message = job_status.get('error', 'Unknown error')
                    logging.error(f"Gladia job failed: {error_message}")
                    return {'text': '', 'error': f"Job failed: {error_message}"}

                time.sleep(self.poll_interval)
            else:
                logging.error("Gladia job timed out")
                return {'text': '', 'error': 'Job timed out'}

            # Extract transcription text and chunks
            text = ''
            chunks = []
            confidence = 0.0

            # v2 shape: result.transcription.{full_transcript, utterances[]}
            result = job_status.get('result', {}) if job_status else {}
            transcription = result.get('transcription', {}) if result else {}

            if transcription:
                text = transcription.get('full_transcript', '') or ''
                utterances = transcription.get('utterances', []) or []

                if not text and utterances:
                    text = ' '.join(u.get('text', '') for u in utterances)

                for u in utterances:
                    for w in u.get('words', []) or []:
                        chunks.append({
                            'word': w.get('word', ''),
                            'start_time': w.get('start', 0),
                            'end_time': w.get('end', 0),
                            'confidence': w.get('confidence', 0),
                            'punctuated_word': w.get('word', '')
                        })

            # Fallbacks for alternative shapes
            if not text:
                text = job_status.get('transcription', '') or job_status.get('text', '') or ''

            if chunks:
                try:
                    confidence = sum(float(c.get('confidence', 0) or 0) for c in chunks) / max(len(chunks), 1)
                except Exception:
                    confidence = 0.0

            logging.info(f"Transcription successful. Text length: {len(text)}, Chunks: {len(chunks)}")

            return {'text': text, 'chunks': chunks, 'confidence': confidence}

        except Exception as e:
            logging.error(f"Error transcribing with Gladia: {e}")
            return {'text': '', 'error': str(e)}