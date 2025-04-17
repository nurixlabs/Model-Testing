import os
import boto3
import json
import csv
import tempfile
import soundfile as sf
from jiwer import wer, cer
from tqdm import tqdm

def list_files_in_s3(bucket_name, prefix, extensions):
    """
    List all files in the specified S3 bucket and prefix matching the given extensions.

    Parameters:
        bucket_name (str): Name of the S3 bucket.
        prefix (str): Prefix within the S3 bucket.
        extensions (tuple): File extensions to filter by.

    Returns:
        List[str]: List of S3 object keys for matching files.
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    files = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.lower().endswith(extensions):
                    files.append(key)
    return files

def download_file_from_s3(bucket_name, object_key, local_dir):
    """
    Downloads a file from S3 to a local directory.

    Parameters:
        bucket_name (str): Name of the S3 bucket.
        object_key (str): Key of the S3 object.
        local_dir (str): Local directory to save the file.

    Returns:
        str: Path to the local file.
    """
    s3 = boto3.client('s3')
    local_filename = os.path.basename(object_key)
    local_path = os.path.join(local_dir, local_filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket_name, object_key, local_path)
    return local_path

def build_transcript_dict(bucket_name, transcript_keys, tmpdir):
    """
    Downloads and parses transcript files from S3 to build a mapping from audio file IDs to transcripts.
    """
    transcript_dict = {}
    for transcript_key in transcript_keys:
        local_transcript_path = download_file_from_s3(bucket_name, transcript_key, tmpdir)
        with open(local_transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    key, transcript = parts
                    transcript_dict[key] = transcript.lower()
    return transcript_dict

def get_audio_duration(audio_path):
    """
    Get the duration of an audio file in seconds.
    """
    with sf.SoundFile(audio_path) as f:
        duration = len(f) / f.samplerate
    return duration

def save_result(output_dir, file_id, result, ground_truth=None):
    """
    Save transcription result as JSON and update CSV.
    
    Parameters:
        output_dir (str): Directory to save results to.
        file_id (str): ID of the audio file.
        result (dict): Transcription result.
        ground_truth (str, optional): Ground truth transcript.
    """
    # Calculate WER and CER if ground truth is available
    if ground_truth and 'text' in result:
        result['ground_truth'] = ground_truth
        result['wer'] = wer(ground_truth.lower(), result['text'].lower())
        result['cer'] = cer(ground_truth.lower(), result['text'].lower())
    
    # Save JSON result
    json_path = os.path.join(output_dir, f"{file_id}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Update CSV
    csv_path = os.path.join(output_dir, 'results.csv')
    csv_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_id', 'ground_truth', 'hypothesis', 'wer', 'cer']
        if 'confidence' in result:
            fieldnames.append('confidence')
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not csv_exists:
            writer.writeheader()
        
        # Prepare row to write
        row = {
            'file_id': file_id,
            'ground_truth': ground_truth if ground_truth else '',
            'hypothesis': result.get('text', ''),
            'wer': result.get('wer', ''),
            'cer': result.get('cer', ''),
        }
        if 'confidence' in result:
            row['confidence'] = result.get('confidence', '')
        
        writer.writerow(row)

def calculate_metrics(results):
    """
    Calculate aggregate metrics from a list of results.
    
    Parameters:
        results (list): List of dictionaries with wer and cer keys.
        
    Returns:
        dict: Aggregated metrics including average WER and CER.
    """
    if not results:
        return {'avg_wer': 0, 'avg_cer': 0, 'num_files': 0}
    
    wer_sum = sum(r.get('wer', 0) for r in results if 'wer' in r)
    cer_sum = sum(r.get('cer', 0) for r in results if 'cer' in r)
    count = len(results)
    
    return {
        'avg_wer': wer_sum / count if count > 0 else 0,
        'avg_cer': cer_sum / count if count > 0 else 0,
        'num_files': count
    }

def prepare_output_dir(base_dir, model_name, test_set):
    """
    Creates and returns the output directory path.
    
    Parameters:
        base_dir (str): Base directory for all results
        model_name (str): Name of the model
        test_set (str): Name of the test set (e.g., 'test-clean')
        
    Returns:
        str: Path to the output directory
    """
    output_dir = os.path.join(base_dir, model_name, test_set)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir