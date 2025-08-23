"""
Azure Speech-to-Text Model Implementation
"""
import os
import time
import json
import logging
import tempfile
import subprocess
import azure.cognitiveservices.speech as speechsdk
from models.base_model import BaseModel



class AzureModel(BaseModel):
    """Azure Speech Services STT implementation."""

    def __init__(self, config):
        super().__init__(config)
        self.name = "azure"
        self.subscription_key = config.get('subscription_key', os.environ.get('AZURE_SPEECH_KEY'))
        self.region = config.get('region', os.environ.get('AZURE_SPEECH_REGION'))
        self.language = config.get('language', 'en-IN')
        self.enable_word_timing = config.get('enable_word_timing', True)
        self.enable_punctuation = config.get('enable_punctuation', True)
        self.use_true_text = config.get('use_true_text', False)  # optional
        self.speech_config = None

        self.language_mapping = {
            'english': 'en-IN', 'en': 'en-IN', 'en-US': 'en-US', 'en-IN': 'en-IN', 'en-GB': 'en-GB',
            'hindi': 'hi-IN', 'hi': 'hi-IN', 'hi-IN': 'hi-IN', 'hinglish': 'hi-IN',
            'marathi': 'mr-IN', 'mr': 'mr-IN', 'mr-IN': 'mr-IN',
        }

    def load(self):
        """Initialize Azure Speech configuration."""
        if not self.subscription_key:
            raise ValueError("Azure subscription key is required. Set it in config or AZURE_SPEECH_KEY.")

        language_code = self._get_language_code()
        logging.info("Initializing Azure Speech client")
        logging.info(f"Region: {self.region}, Language: {language_code}")

        self.speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
        self.speech_config.speech_recognition_language = language_code

        # Get detailed JSON so evt.result.json contains NBest/Words
        self.speech_config.output_format = speechsdk.OutputFormat.Detailed  # enables detailed result payload
        # Request word-level timestamps (Words array)
        if self.enable_word_timing:
            self.speech_config.request_word_level_timestamps()  # includes per-word offset/duration

        # Auto-punct is on by default; enable dictation if you want spoken punctuation ("comma", "question mark")
        if self.enable_punctuation:
            self.speech_config.enable_dictation()

        # Optional: normalize disfluencies (TrueText)
        if self.use_true_text:
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption, "TrueText"
            )

        logging.info("Azure Speech client initialized successfully")

    def _get_language_code(self, language=None):
        lang = language or self.language
        mapped = self.language_mapping.get(lang, lang)
        if '-' not in mapped:
            logging.warning(f"Invalid language code '{mapped}' for Azure, defaulting to 'en-IN'")
            return 'en-IN'
        return mapped
    
    def _convert_to_azure_format(self, audio_path):
        """
        Convert audio to Azure-compatible WAV format (16kHz, 16-bit, mono PCM).
        Returns path to converted file or original if conversion fails.
        """
        try:
            # Create a temporary file for the converted audio
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            
            # Use ffmpeg to convert to Azure-compatible format
            # -ac 1: mono
            # -ar 16000: 16kHz sample rate
            # -sample_fmt s16: 16-bit signed PCM
            # -map_metadata -1: strip metadata chunks that might cause issues
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ac', '1',
                '-ar', '16000',
                '-sample_fmt', 's16',
                '-map_metadata', '-1',
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logging.info(f"Converted audio to Azure-compatible format: {temp_path}")
                return temp_path
            else:
                logging.warning(f"ffmpeg conversion failed: {result.stderr}")
                os.remove(temp_path)
                return audio_path
                
        except FileNotFoundError:
            logging.warning("ffmpeg not found, using original audio file")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return audio_path
        except Exception as e:
            logging.warning(f"Audio conversion failed: {e}, using original file")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return audio_path

    def transcribe(self, audio_path, language=None):
        """
        Transcribe audio using Azure Speech Services (file input).
        Returns: dict with 'text', 'chunks' (word timings), 'language_used' or 'error'
        """
        converted_audio = None
        try:
            # Convert audio to Azure-compatible format
            converted_audio = self._convert_to_azure_format(audio_path)
            audio_config = speechsdk.audio.AudioConfig(filename=converted_audio)
            
            # For Hinglish, use AutoDetectSourceLanguageConfig for segment-level detection
            if language == 'hinglish' or (not language and self.language == 'hinglish'):
                logging.info("Using AutoDetectSourceLanguageConfig for Hinglish with en-IN and hi-IN")
                auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                    languages=["en-IN", "hi-IN"]
                )
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config, 
                    audio_config=audio_config,
                    auto_detect_source_language_config=auto_detect_config
                )
            else:
                # For other languages, use standard configuration
                if language:
                    language_code = self._get_language_code(language)
                    self.speech_config.speech_recognition_language = language_code
                    logging.info(f"Using language override: {language_code}")
                recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)

            final_text = []
            chunks = []
            done = False

            def on_recognized(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    final_text.append(evt.result.text)
                    # parse detailed JSON for per-word timings
                    if getattr(evt.result, "json", None):
                        try:
                            payload = json.loads(evt.result.json)
                            nbest = payload.get("NBest") or []
                            if nbest and "Words" in nbest[0]:
                                for w in nbest[0]["Words"]:
                                    start = (w.get("Offset", 0)) / 1e7
                                    dur = (w.get("Duration", 0)) / 1e7
                                    chunks.append({
                                        "word": w.get("Word", ""),
                                        "start_time": start,
                                        "end_time": start + dur,
                                        "confidence": w.get("Confidence", 0.0),
                                    })
                        except Exception:
                            pass

            def on_canceled(evt):
                if evt.result.reason == speechsdk.ResultReason.Canceled:
                    details = evt.result.cancellation_details
                    logging.error(f"Azure transcription canceled: {details.reason}")
                    if details.error_details:
                        logging.error(f"Error details: {details.error_details}")

            def on_session_stopped(evt):
                nonlocal done
                done = True

            recognizer.recognized.connect(on_recognized)
            recognizer.canceled.connect(on_canceled)
            recognizer.session_stopped.connect(on_session_stopped)

            logging.info("Starting Azure transcription...")
            recognizer.start_continuous_recognition()

            # wait for file to be processed
            while not done:
                time.sleep(0.1)

            recognizer.stop_continuous_recognition()

            # Determine language used
            lang_used = self.speech_config.speech_recognition_language
            if language == 'hinglish' or (not language and self.language == 'hinglish'):
                lang_used = "hi-IN/en-IN (auto-detect)"
            
            return {
                "text": " ".join(t.strip() for t in final_text).strip(),
                "chunks": chunks,
                "language_used": lang_used,
            }

        except Exception as e:
            logging.error(f"Error transcribing with Azure: {e}")
            return {"text": "", "error": str(e)}
        finally:
            # Clean up temporary converted audio file
            if converted_audio and converted_audio != audio_path and os.path.exists(converted_audio):
                try:
                    os.remove(converted_audio)
                    logging.info(f"Cleaned up temporary audio file: {converted_audio}")
                except Exception as e:
                    logging.warning(f"Failed to clean up temporary file: {e}")
        
# if __name__ == "__main__":
#     import argparse
#     import logging
#     from dotenv import load_dotenv
#     load_dotenv(dotenv_path='/Users/administrator/Desktop/Model-Testing/.env')
#     parser = argparse.ArgumentParser(description="Test AzureModel on a single audio file")
#     parser.add_argument("audio_path", help="Path to the audio file to transcribe")
#     parser.add_argument("--key", help="Azure Speech subscription key (optional)")
#     parser.add_argument("--region", help="Azure Speech region (optional)")
#     parser.add_argument("--language", help="Language code (like en-IN)", default=None)
#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO)

#     # Only add config values if they are provided as arguments
#     config = {}
#     if args.key:
#         config["subscription_key"] = args.key
#     if args.region:
#         config["region"] = args.region
#     if args.language:
#         config["language"] = args.language
#     model = AzureModel(config)

#     try:
#         model.load()
#     except Exception as e:
#         logging.error(f"Failed to initialize AzureModel: {e}")
#         exit(1)

#     logging.info(f"Transcribing file: {args.audio_path}")
#     result = model.transcribe(args.audio_path, language=args.language)

#     if result.get("error"):
#         logging.error(f"Transcription failed: {result['error']}")
#     else:
#         print("\n===== TRANSCRIPTION RESULT =====\n")
#         print(f"Language Used: {result.get('language_used', 'N/A')}\n")
#         print("Transcribed Text:\n")
#         print(result.get("text", ""))

#         chunks = result.get("chunks", [])
#         if chunks:
#             print("\nWord-Level Timings (first 10 words):")
#             for w in chunks[:10]:
#                 print(f"{w['word']} — start: {w['start_time']:.2f}s, end: {w['end_time']:.2f}s, confidence: {w.get('confidence', 'N/A')}")

#     logging.info("Done.")