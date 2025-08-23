"""
AWS Transcribe Speech-to-Text Model Implementation
"""
import os
import boto3
import json
import uuid
import time
import tempfile
import logging
from models.base_model import BaseModel
from botocore.exceptions import ClientError


class AWSModel(BaseModel):
    """AWS Transcribe speech-to-text implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "aws"
        self.region = config.get('region', 'us-east-1')
        self.s3_region = config.get('s3_region', 'us-east-1')
        self.transcribe_client = None
        self.s3_client = None
        self.language_code = config.get('language_code', 'en-US')
        self.language = config.get('language', 'english')
        self.max_concurrent_jobs = config.get('max_concurrent_jobs', 90)
        self.output_bucket_name = config.get('output_bucket_name')
        self.output_prefix = config.get('output_prefix', 'transcripts')
    
    def load(self):
        """Initialize AWS clients."""
        logging.info("Initializing AWS Transcribe and S3 clients")
        logging.info(f"Region: {self.region}, Language: {self.language_code}")
        
        try:
            self.transcribe_client = boto3.client('transcribe', region_name=self.region)
            self.s3_client = boto3.client('s3', region_name=self.s3_region)
            logging.info("AWS clients initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize AWS clients: {e}")
            raise
    
    def transcribe(self, audio_path, bucket_name=None, audio_key=None):
        """
        Transcribe audio using AWS Transcribe.
        
        Args:
            audio_path: Path to audio file
            bucket_name: S3 bucket name (optional)
            audio_key: S3 key (optional)
            
        Returns:
            dict: Transcription results
        """
        try:
            # Generate unique job name
            job_name = f"transcribe-{str(uuid.uuid4())}"
            
            # Determine media format
            file_ext = os.path.splitext(audio_path)[1].lower()
            media_format_map = {
                '.wav': 'wav', '.mp3': 'mp3', '.flac': 'flac', 
                '.ogg': 'ogg', '.amr': 'amr', '.webm': 'webm',
                '.mp4': 'mp4', '.m4a': 'm4a'
            }
            media_format = media_format_map.get(file_ext, 'mp3')
            
            # Handle S3 upload if needed
            if not bucket_name:
                bucket_name = "mlflow-artifacts-nurix"
                audio_key = f"aws-results/temp-uploads/{os.path.basename(audio_path)}"
                
                logging.info(f"Uploading {audio_path} to s3://{bucket_name}/{audio_key}")
                self.s3_client.upload_file(audio_path, bucket_name, audio_key)
            
            # Configure output location
            self.output_bucket_name = bucket_name
            output_key = f"{self.output_prefix}/{job_name}/transcript.json"
            
            # Submit transcription job
            # For Hinglish, use language identification
            if self.language == 'hinglish':
                response = self.transcribe_client.start_transcription_job(
                    TranscriptionJobName=job_name,
                    Media={'MediaFileUri': f"s3://{bucket_name}/{audio_key}"},
                    MediaFormat=media_format,
                    IdentifyLanguage=True,
                    LanguageOptions=['en-IN', 'hi-IN'],  # Support both languages for code-switching
                    OutputBucketName=bucket_name,
                    OutputKey=output_key
                )
                logging.info(f"Submitted Hinglish transcription job with language identification: en-IN, hi-IN")
            else:
                response = self.transcribe_client.start_transcription_job(
                    TranscriptionJobName=job_name,
                    Media={'MediaFileUri': f"s3://{bucket_name}/{audio_key}"},
                    MediaFormat=media_format,
                    LanguageCode=self.language_code,
                    OutputBucketName=bucket_name,
                    OutputKey=output_key
                )
            
            logging.info(f"Submitted transcription job {job_name}, waiting for completion")
            
            # Wait for job completion
            while True:
                status = self.transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                
                job_status = status['TranscriptionJob']['TranscriptionJobStatus']
                
                if job_status in ['COMPLETED', 'FAILED']:
                    break
                    
                time.sleep(5)
            
            if job_status == 'FAILED':
                error_reason = status['TranscriptionJob'].get('FailureReason', 'Unknown error')
                logging.error(f"AWS Transcription job failed: {error_reason}")
                return {
                    'text': '',
                    'error': f'Transcription job failed: {error_reason}'
                }
            
            # Download and parse results
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            self.s3_client.download_file(bucket_name, output_key, tmp_path)
            
            with open(tmp_path, 'r', encoding='utf-8') as f:
                transcript_result = json.load(f)
            
            os.remove(tmp_path)
            
            # Extract transcript and word information
            transcript_text = transcript_result.get('results', {}).get('transcripts', [{}])[0].get('transcript', '')
            items = transcript_result.get('results', {}).get('items', [])
            
            chunks = []
            for item in items:
                if item.get('type') == 'pronunciation':
                    alternative = item.get('alternatives', [{}])[0]
                    chunks.append({
                        'word': alternative.get('content', ''),
                        'start_time': float(item.get('start_time', 0)),
                        'end_time': float(item.get('end_time', 0)),
                        'confidence': float(alternative.get('confidence', 0)),
                        'punctuated_word': alternative.get('content', '')
                    })
            
            # Clean up temporary S3 objects
            if not audio_key or not bucket_name:
                try:
                    self.s3_client.delete_object(Bucket=bucket_name, Key=audio_key)
                    self.s3_client.delete_object(Bucket=bucket_name, Key=output_key)
                except Exception as e:
                    logging.warning(f"Failed to clean up S3 objects: {e}")
            
            # Calculate average confidence
            confidence = sum(chunk.get('confidence', 0) for chunk in chunks) / len(chunks) if chunks else 0
            
            return {
                'text': transcript_text,
                'chunks': chunks,
                'confidence': confidence
            }
            
        except Exception as e:
            logging.error(f"Error transcribing with AWS: {e}")
            return {
                'text': '',
                'error': str(e)
            }