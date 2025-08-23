#!/usr/bin/env python3
"""
LibriSpeech Dataset Processing Pipeline
Handles test-clean and test-other datasets from S3
"""
import os
import json
import tempfile
import logging
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from config import S3_CONFIG, OUTPUT_CONFIG
from utils import (
    list_files_in_s3,
    download_file_from_s3,
    build_transcript_dict,
    get_audio_duration,
    save_result,
    prepare_output_dir,
    calculate_metrics
)
from models.model_factory import get_model


def process_test_set(
    model: Any,
    bucket_name: str,
    prefix: str,
    output_dir: str,
    test_set: str
) -> Optional[Dict[str, Any]]:
    """
    Process a LibriSpeech test set (test-clean or test-other).
    
    Args:
        model: Loaded STT model instance
        bucket_name: S3 bucket name
        prefix: S3 prefix for the test set
        output_dir: Base output directory
        test_set: Name of the test set
        
    Returns:
        dict: Metrics for this test set or None if no files found
    """
    logging.info(f"Processing dataset: {test_set} from s3://{bucket_name}/{prefix}")
    
    # Create output directory for this test set
    test_output_dir = prepare_output_dir(output_dir, model.name, test_set)
    
    # List audio and transcript files
    audio_files = list_files_in_s3(bucket_name, prefix, ('.wav', '.mp3', '.flac'))
    transcript_files = list_files_in_s3(bucket_name, prefix, ('.trans.txt',))
    
    if not audio_files:
        logging.warning(f"No audio files found in s3://{bucket_name}/{prefix}")
        return None
    
    if not transcript_files:
        logging.warning(f"No transcript files found in s3://{bucket_name}/{prefix}")
        return None
    
    logging.info(f"Found {len(audio_files)} audio file(s) and {len(transcript_files)} transcript file(s)")
    
    # Initialize CSV file
    csv_path = os.path.join(test_output_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvfile.write('file_id,ground_truth,hypothesis,wer,cer\n')
    
    # Process files in temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        logging.debug(f"Created temporary directory: {tmpdir}")
        
        # Build transcript dictionary
        logging.info("Building transcript dictionary...")
        transcript_dict = build_transcript_dict(bucket_name, transcript_files, tmpdir)
        
        # Initialize metrics
        total_duration = 0.0
        results = []
        error_count = 0
        max_errors = 5
        
        # Process each audio file
        for audio_file_key in tqdm(audio_files, desc=f"Processing {test_set}"):
            if total_duration >= OUTPUT_CONFIG['max_audio_duration']:
                logging.info(f"Reached maximum audio duration limit ({OUTPUT_CONFIG['max_audio_duration']}s)")
                break
            
            # Extract file ID
            file_id = os.path.splitext(os.path.basename(audio_file_key))[0]
            
            # Skip if no ground truth
            if file_id not in transcript_dict:
                logging.debug(f"No ground truth found for {file_id}, skipping")
                continue
            
            try:
                # Download audio file
                local_audio_path = download_file_from_s3(bucket_name, audio_file_key, tmpdir)
                
                # Get audio duration
                duration = get_audio_duration(local_audio_path)
                total_duration += duration
                
                # Get ground truth
                ground_truth = transcript_dict[file_id]
                
                # Transcribe
                logging.debug(f"Transcribing {file_id}...")
                result = model.transcribe(local_audio_path)
                
                # Save result
                save_result(test_output_dir, file_id, result, ground_truth)
                
                # Track metrics
                if 'wer' in result and 'cer' in result:
                    results.append({
                        'file_id': file_id,
                        'wer': result['wer'],
                        'cer': result['cer']
                    })
                
            except Exception as e:
                logging.error(f"Error processing {file_id}: {e}")
                error_count += 1
                if error_count >= max_errors:
                    logging.error(f"Too many errors ({error_count}), stopping")
                    break
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        # Convert duration to hours
        total_hours = total_duration / 3600.0
        
        # Save metrics
        metrics_path = os.path.join(test_output_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'test_set': test_set,
                'model': model.name,
                'num_files': metrics['num_files'],
                'total_duration_seconds': total_duration,
                'total_duration_hours': total_hours,
                'avg_wer': metrics['avg_wer'],
                'avg_cer': metrics['avg_cer']
            }, f, indent=2)
        
        # Log results
        logging.info(f"\n{test_set} Results:")
        logging.info(f"Files processed: {metrics['num_files']}")
        logging.info(f"Total duration: {total_duration:.2f}s ({total_hours:.2f}h)")
        logging.info(f"Average WER: {metrics['avg_wer']:.4f}")
        logging.info(f"Average CER: {metrics['avg_cer']:.4f}")
        
        return {
            'num_files': metrics['num_files'],
            'total_duration': total_duration,
            'avg_wer': metrics['avg_wer'],
            'avg_cer': metrics['avg_cer']
        }


def process_librispeech(
    model_name: str,
    model_config: Dict[str, Any],
    test_set: str = 'both',
    output_dir: str = None
) -> None:
    """
    Main entry point for LibriSpeech processing.
    
    Args:
        model_name: Name of the model to use
        model_config: Model configuration dictionary
        test_set: Which test set(s) to process ('test-clean', 'test-other', 'both')
        output_dir: Output directory path
    """
    # Set default output directory
    if output_dir is None:
        output_dir = OUTPUT_CONFIG['base_dir']
    
    # Create and load model
    logging.info(f"Initializing {model_name} model...")
    model = get_model(model_name, model_config)
    model.load()
    
    # Process test sets
    results = {}
    
    if test_set in ['test-clean', 'both']:
        logging.info("\nProcessing test-clean dataset...")
        test_clean_metrics = process_test_set(
            model,
            S3_CONFIG['bucket_name'],
            S3_CONFIG['test_clean_prefix'],
            output_dir,
            'test-clean'
        )
        if test_clean_metrics:
            results['test-clean'] = test_clean_metrics
    
    if test_set in ['test-other', 'both']:
        logging.info("\nProcessing test-other dataset...")
        test_other_metrics = process_test_set(
            model,
            S3_CONFIG['bucket_name'],
            S3_CONFIG['test_other_prefix'],
            output_dir,
            'test-other'
        )
        if test_other_metrics:
            results['test-other'] = test_other_metrics
    
    # Save overall results
    if results:
        overall_metrics_path = os.path.join(output_dir, f"{model_name}_librispeech_metrics.json")
        with open(overall_metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': model_name,
                'dataset': 'librispeech',
                'results': results
            }, f, indent=2)
        
        logging.info("\n=== OVERALL RESULTS ===")
        for test_set_name, metrics in results.items():
            logging.info(
                f"{test_set_name}: WER={metrics['avg_wer']:.4f}, "
                f"CER={metrics['avg_cer']:.4f}, Files={metrics['num_files']}"
            )


# ==============================================================================
# Flexible dataset entry points for English
# ==============================================================================

# === Unified dataset options ===
DATASET_CONFIGS = {
    'librispeech': {
        'bucket_name': S3_CONFIG['bucket_name'],
        'prefixes': {
            'test-clean': S3_CONFIG['test_clean_prefix'],
            'test-other': S3_CONFIG['test_other_prefix'],
        },
        'language': 'english',
    },
    'marathi-asr': {
        'source': 'huggingface',
        'dataset_name': 'TheAIchemist13/marathi_asr_dataset',
        'language': 'marathi',
    },
    'custom-csv': {
        'source': 'csv',
        'language': 'hinglish',  # Can be overridden
    },
}


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Parse an S3 URI and return (bucket_name, key). If not an s3 uri, returns (None, path).
    """
    if not isinstance(s3_uri, str):
        return None, s3_uri
    if not s3_uri.startswith('s3://'):
        # If it's already just a key path or local path, return None for bucket
        return None, s3_uri
    path = s3_uri[5:]
    parts = path.split('/', 1)
    if len(parts) == 2:
        bucket_name, key = parts
        return bucket_name, key
    else:
        return parts[0], ''


def _english_iter_hf_samples(dataset_name: str, split: str, tmpdir: str, audio_col: str = 'audio', text_col: str = 'text'):
    from datasets import load_dataset
    import soundfile as sf
    ds = load_dataset(dataset_name)
    test_ds = ds[split] if split in ds else ds['test']
    for idx, sample in enumerate(test_ds):
        audio = sample.get(audio_col)
        text = sample.get(text_col) or sample.get('transcription') or sample.get('text')
        if audio is None or text is None:
            continue
        local_path = None
        # HuggingFace Audio object or dict with array
        try:
            if isinstance(audio, dict) and 'array' in audio and 'sampling_rate' in audio:
                local_path = os.path.join(tmpdir, f"{idx}.wav")
                sf.write(local_path, audio['array'], audio['sampling_rate'])
            elif hasattr(audio, 'array') and hasattr(audio, 'sampling_rate'):
                local_path = os.path.join(tmpdir, f"{idx}.wav")
                sf.write(local_path, audio.array, audio.sampling_rate)
            elif isinstance(audio, str):
                # path string (local or s3)
                b, k = parse_s3_uri(audio)
                if b:
                    local_path = download_file_from_s3(b, k, tmpdir)
                else:
                    local_path = audio
            else:
                continue
        except Exception:
            continue
        file_id = os.path.splitext(os.path.basename(local_path))[0]
        yield file_id, local_path, text


def _english_iter_csv_samples(csv_path: str, tmpdir: str, default_bucket: str):
    import csv
    with open(csv_path, encoding='utf-8') as in_csv:
        reader = csv.DictReader(in_csv)
        for row in reader:
            s3_path = row.get('Key') or row.get('audio_path')
            if not s3_path:
                continue
            gb = row.get('ground_truth') or row.get('Transcription')
            if not gb:
                continue
            b, k = parse_s3_uri(s3_path)
            target_bucket = b if b else default_bucket
            local_path = download_file_from_s3(target_bucket, k, tmpdir) if target_bucket else s3_path
            file_id = os.path.splitext(os.path.basename(local_path))[0]
            yield file_id, local_path, gb


def process_english(
    model_name: str,
    model_config: Dict[str, Any],
    dataset_key: str = 'librispeech',
    output_dir: Optional[str] = None,
    test_set: str = 'both',
    hf_dataset_name: Optional[str] = None,
    hf_split: str = 'test',
    csv_path: Optional[str] = None,
) -> None:
    """
    Unified English entrypoint. Supports:
      • LibriSpeech from S3 (default)
      • Any HuggingFace dataset with audio/text columns
      • A custom CSV with columns: Key|audio_path, ground_truth|Transcription
    """
    if output_dir is None:
        output_dir = OUTPUT_CONFIG['base_dir']

    # LibriSpeech default path
    if dataset_key == 'librispeech' and not (hf_dataset_name or csv_path):
        return process_librispeech(model_name, model_config, test_set=test_set, output_dir=output_dir)

    # Prepare model
    logging.info(f"Initializing {model_name} model...")
    model = get_model(model_name, model_config)
    model.load()

    # Decide dataset type
    cfg = DATASET_CONFIGS.get(dataset_key, {})
    source = cfg.get('source')
    
    if csv_path or source == 'csv':
        # CSV processing path
        dataset_name = os.path.splitext(os.path.basename(csv_path or 'custom.csv'))[0]
        test_output_dir = prepare_output_dir(output_dir, model_name, dataset_name)
        results_csv = os.path.join(test_output_dir, 'results.csv')
        os.makedirs(test_output_dir, exist_ok=True)
        
        with open(results_csv, 'w', newline='', encoding='utf-8') as csvfile:
            csvfile.write('file_id,ground_truth,hypothesis,wer,cer\n')
        
        total_duration = 0.0
        results = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            default_bucket = S3_CONFIG.get('bucket_name')
            for file_id, local_path, ground_truth in _english_iter_csv_samples(csv_path, tmpdir, default_bucket):
                if total_duration >= OUTPUT_CONFIG['max_audio_duration']:
                    break
                try:
                    duration = get_audio_duration(local_path)
                    total_duration += duration
                    result = model.transcribe(local_path)
                    save_result(test_output_dir, file_id, result, ground_truth)
                    if 'wer' in result and 'cer' in result:
                        results.append({'file_id': file_id, 'wer': result['wer'], 'cer': result['cer']})
                except Exception as e:
                    logging.error(f"Error processing {file_id}: {e}")
        
        # Calculate and save metrics
        if results:
            metrics = calculate_metrics(results)
            metrics_path = os.path.join(test_output_dir, 'metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'dataset': dataset_name,
                    'model': model_name,
                    'avg_wer': metrics['avg_wer'],
                    'avg_cer': metrics['avg_cer'],
                    'num_files': metrics['num_files'],
                    'total_duration': total_duration
                }, f, indent=2)
            logging.info(f"Metrics saved to {metrics_path}")
            logging.info(f"Average WER: {metrics['avg_wer']:.4f}, Average CER: {metrics['avg_cer']:.4f}")
        return

    # HuggingFace path
    dataset_name = hf_dataset_name or cfg.get('dataset_name')
    if not dataset_name:
        raise ValueError("Please provide hf_dataset_name or use a DATASET_CONFIGS key with dataset_name.")
    
    dataset_tag = dataset_name.split('/')[-1]
    test_output_dir = prepare_output_dir(output_dir, model_name, f"hf-{dataset_tag}")
    results_csv = os.path.join(test_output_dir, 'results.csv')
    os.makedirs(test_output_dir, exist_ok=True)
    
    with open(results_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvfile.write('file_id,ground_truth,hypothesis,wer,cer\n')
    
    total_duration = 0.0
    results = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for file_id, local_path, ground_truth in _english_iter_hf_samples(dataset_name, hf_split, tmpdir):
            if total_duration >= OUTPUT_CONFIG['max_audio_duration']:
                break
            try:
                duration = get_audio_duration(local_path)
                total_duration += duration
                result = model.transcribe(local_path)
                save_result(test_output_dir, file_id, result, ground_truth)
                if 'wer' in result and 'cer' in result:
                    results.append({'file_id': file_id, 'wer': result['wer'], 'cer': result['cer']})
            except Exception as e:
                logging.error(f"Error processing {file_id}: {e}")
    
    # Calculate and save metrics
    if results:
        metrics = calculate_metrics(results)
        metrics_path = os.path.join(test_output_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset_tag,
                'model': model_name,
                'avg_wer': metrics['avg_wer'],
                'avg_cer': metrics['avg_cer'],
                'num_files': metrics['num_files'],
                'total_duration': total_duration
            }, f, indent=2)
        logging.info(f"Metrics saved to {metrics_path}")
        logging.info(f"Average WER: {metrics['avg_wer']:.4f}, Average CER: {metrics['avg_cer']:.4f}")
    return