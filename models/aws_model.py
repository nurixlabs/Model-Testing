import os
import boto3
import json
import uuid
import time
import tempfile
from models.base_model import BaseModel
from botocore.exceptions import ClientError

class AWSModel(BaseModel):
    """AWS Transcribe speech-to-text implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "aws"
        self.transcribe_client = None
        self.s3_client = None
        self.language_code = config.get('language_code', 'en-US')
        self.max_concurrent_jobs = config.get('max_concurrent_jobs', 90)
        self.output_bucket_name = config.get('output_bucket_name')  # Will be set during transcription
        self.output_prefix = config.get('output_prefix', 'transcripts')
    
    def load(self):
        """Initialize AWS clients."""
        print("Initializing AWS Transcribe and S3 clients...")
        self.transcribe_client = boto3.client('transcribe')
        self.s3_client = boto3.client('s3')
        print("AWS clients initialized successfully.")
    
    def transcribe(self, audio_path, bucket_name=None, audio_key=None):
        """
        Transcribe the audio file using AWS Transcribe.
        
        This is a synchronous wrapper around the asynchronous AWS Transcribe API.
        It uploads the file to S3 (if not already there), submits a job, waits for
        completion, and then downloads and parses the results.
        
        Args:
            audio_path (str): Path to the audio file
            bucket_name (str, optional): S3 bucket name if file is already in S3
            audio_key (str, optional): S3 key if file is already in S3
            
        Returns:
            dict: Dictionary containing:
                - text (str): The transcribed text
                - chunks (list): Word-level information with timing
        """
        try:
            # Generate a unique job name
            job_name = f"transcribe-{str(uuid.uuid4())}"
            
            # Determine file format from extension
            file_ext = os.path.splitext(audio_path)[1].lower()
            media_format_map = {
                '.wav': 'wav', '.mp3': 'mp3', '.flac': 'flac', 
                '.ogg': 'ogg', '.amr': 'amr', '.webm': 'webm',
                '.mp4': 'mp4', '.m4a': 'm4a'
            }
            media_format = media_format_map.get(file_ext, 'mp3')  # Default to mp3 if unknown
            
            # If no bucket name provided, use a temporary one
            # For a real implementation, you would want to upload to a specific bucket
            if not bucket_name:
                bucket_name = "your-transcribe-temp-bucket"  # Replace with your bucket
                
                # Generate a temporary S3 key
                audio_key = f"temp-uploads/{os.path.basename(audio_path)}"
                
                # Upload file to S3
                print(f"Uploading {audio_path} to s3://{bucket_name}/{audio_key}...")
                self.s3_client.upload_file(audio_path, bucket_name, audio_key)
            
            # Configure the output location
            self.output_bucket_name = bucket_name
            output_key = f"{self.output_prefix}/{job_name}/transcript.json"
            
            # Submit the transcription job
            response = self.transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={
                    'MediaFileUri': f"s3://{bucket_name}/{audio_key}"
                },
                MediaFormat=media_format,
                LanguageCode=self.language_code,
                OutputBucketName=bucket_name,
                OutputKey=output_key,
                Settings={
                    'ShowSpeakerLabels': True,
                    'MaxSpeakerLabels': 2,
                    'ShowAlternatives': True,
                    'MaxAlternatives': 2
                }
            )
            
            print(f"Submitted transcription job {job_name}, waiting for completion...")
            
            # Wait for job to complete
            while True:
                status = self.transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                
                job_status = status['TranscriptionJob']['TranscriptionJobStatus']
                
                if job_status in ['COMPLETED', 'FAILED']:
                    break
                    
                print(f"Job status: {job_status}, waiting...")
                time.sleep(5)
            
            if job_status == 'FAILED':
                error_reason = status['TranscriptionJob'].get('FailureReason', 'Unknown error')
                raise Exception(f"Transcription job failed: {error_reason}")
            
            # Download the transcript result
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            self.s3_client.download_file(bucket_name, output_key, tmp_path)
            
            # Parse the result
            with open(tmp_path, 'r', encoding='utf-8') as f:
                transcript_result = json.load(f)
            
            # Clean up temp file
            os.remove(tmp_path)
            
            # Extract text and word information
            transcript_text = transcript_result.get('results', {}).get('transcripts', [{}])[0].get('transcript', '')
            
            # Extract word information
            items = transcript_result.get('results', {}).get('items', [])
            word_segments = []
            
            for item in items:
                if item.get('type') == 'pronunciation':
                    start_time = float(item.get('start_time', 0))
                    end_time = float(item.get('end_time', 0))
                    content = item.get('alternatives', [{}])[0].get('content', '')
                    confidence = float(item.get('alternatives', [{}])[0].get('confidence', 0))
                    
                    word_segments.append({
                        'word': content,
                        'start_time': start_time,
                        'end_time': end_time,
                        'confidence': confidence
                    })
            
            # Clean up S3 if we uploaded temporarily
            if not audio_key or not bucket_name:
                try:
                    self.s3_client.delete_object(Bucket=bucket_name, Key=audio_key)
                    self.s3_client.delete_object(Bucket=bucket_name, Key=output_key)
                except Exception as e:
                    print(f"Warning: Failed to clean up S3 objects: {e}")
            
            return {
                'text': transcript_text,
                'chunks': word_segments,
                'job_name': job_name,
                'confidence': sum(segment.get('confidence', 0) for segment in word_segments) / len(word_segments) if word_segments else 0
            }
        
        except Exception as e:
            print(f"Error in AWS transcription: {e}")
            return {'text': '', 'error': str(e)}