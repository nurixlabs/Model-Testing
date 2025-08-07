"""
Utility functions for the STT evaluation pipeline
"""
import os
import csv
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import boto3
import soundfile as sf
from jiwer import wer, cer


# Initialize S3 client
s3_client = boto3.client('s3')


def list_files_in_s3(bucket_name: str, prefix: str, extensions: Tuple[str, ...]) -> List[str]:
    """
    List files in S3 bucket with specific extensions.
    
    Args:
        bucket_name: S3 bucket name
        prefix: S3 prefix to search in
        extensions: Tuple of file extensions to filter
        
    Returns:
        List of S3 keys matching the extensions
    """
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    try:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.lower().endswith(extensions):
                        files.append(key)
    except Exception as e:
        logging.error(f"Error listing files in s3://{bucket_name}/{prefix}: {e}")
    
    return sorted(files)


def download_file_from_s3(bucket_name: str, key: str, local_dir: str) -> str:
    """
    Download a file from S3 to local directory.
    
    Args:
        bucket_name: S3 bucket name
        key: S3 object key
        local_dir: Local directory to save the file
        
    Returns:
        Path to the downloaded file
    """
    filename = os.path.basename(key)
    local_path = os.path.join(local_dir, filename)
    
    try:
        logging.debug(f"Downloading s3://{bucket_name}/{key} to {local_path}")
        s3_client.download_file(bucket_name, key, local_path)
        return local_path
    except Exception as e:
        logging.error(f"Error downloading {key}: {e}")
        raise


def build_transcript_dict(
    bucket_name: str, 
    transcript_files: List[str], 
    local_dir: str
) -> Dict[str, str]:
    """
    Build a dictionary mapping file IDs to transcripts.
    
    Args:
        bucket_name: S3 bucket name
        transcript_files: List of transcript file keys
        local_dir: Local directory for downloads
        
    Returns:
        Dictionary mapping file_id to transcript text
    """
    transcript_dict = {}
    
    for trans_file in transcript_files:
        try:
            # Download transcript file
            local_path = download_file_from_s3(bucket_name, trans_file, local_dir)
            
            # Read and parse transcripts
            with open(local_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            file_id, transcript = parts
                            transcript_dict[file_id] = transcript
                        else:
                            logging.warning(f"Invalid transcript line: {line}")
        except Exception as e:
            logging.error(f"Error processing transcript file {trans_file}: {e}")
    
    return transcript_dict


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of an audio file in seconds.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception as e:
        logging.error(f"Error getting duration for {audio_path}: {e}")
        return 0.0


def save_result(
    output_dir: str, 
    file_id: str, 
    result: Dict[str, Any], 
    ground_truth: str
) -> None:
    """
    Save transcription result to files.
    
    Args:
        output_dir: Output directory
        file_id: File identifier
        result: Transcription result dictionary
        ground_truth: Ground truth transcript
    """
    # Extract hypothesis text
    hypothesis = result.get('text', '')
    
    # Calculate WER and CER
    try:
        wer_score = wer(ground_truth, hypothesis)
        cer_score = cer(ground_truth, hypothesis)
    except:
        wer_score = 1.0
        cer_score = 1.0
    
    # Add metrics to result
    result['wer'] = wer_score
    result['cer'] = cer_score
    
    # Save individual result as JSON
    result_path = os.path.join(output_dir, f"{file_id}.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'file_id': file_id,
            'ground_truth': ground_truth,
            'hypothesis': hypothesis,
            'wer': wer_score,
            'cer': cer_score,
            'chunks': result.get('chunks', []),
            'confidence': result.get('confidence', 0),
            'error': result.get('error', None)
        }, f, indent=2)
    
    # Append to CSV
    csv_path = os.path.join(output_dir, 'results.csv')
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([file_id, ground_truth, hypothesis, wer_score, cer_score])


def prepare_output_dir(base_dir: str, model_name: str, dataset_name: str) -> str:
    """
    Prepare output directory structure.
    
    Args:
        base_dir: Base output directory
        model_name: Name of the model
        dataset_name: Name of the dataset
        
    Returns:
        Path to the created directory
    """
    output_dir = os.path.join(base_dir, model_name, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    logging.debug(f"Created output directory: {output_dir}")
    return output_dir


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate average metrics from results.
    
    Args:
        results: List of result dictionaries with 'wer' and 'cer' keys
        
    Returns:
        Dictionary with aggregated metrics
    """
    if not results:
        return {
            'num_files': 0,
            'avg_wer': 0.0,
            'avg_cer': 0.0
        }
    
    num_files = len(results)
    avg_wer = sum(r.get('wer', 0) for r in results) / num_files
    avg_cer = sum(r.get('cer', 0) for r in results) / num_files
    
    return {
        'num_files': num_files,
        'avg_wer': avg_wer,
        'avg_cer': avg_cer
    }