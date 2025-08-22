#!/usr/bin/env python3
"""
Marathi ASR Dataset Processing Pipeline
Handles the HuggingFace Marathi ASR dataset
"""
import os
import json
import tempfile
import logging
from typing import Dict, Any, Optional
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset

from config import OUTPUT_CONFIG
from utils import (
    get_audio_duration,
    save_result,
    prepare_output_dir,
    calculate_metrics
)
from models.model_factory import get_model

from config import S3_CONFIG, OUTPUT_CONFIG

def process_marathi_asr(
    model_name: str,
    model_config: Dict[str, Any],
    output_dir: str = None
) -> None:
    """
    Process the Marathi ASR dataset from HuggingFace.
    
    Args:
        model_name: Name of the model to use
        model_config: Model configuration dictionary
        output_dir: Output directory path
    """
    # Set default output directory
    if output_dir is None:
        output_dir = OUTPUT_CONFIG['base_dir']
    
    # Create and load model
    logging.info(f"Initializing {model_name} model...")
    model = get_model(model_name, model_config)
    model.load()
    
    # Load dataset
    logging.info("Loading Marathi ASR dataset from HuggingFace...")
    try:
        dataset = load_dataset("TheAIchemist13/marathi_asr_dataset")
        test_dataset = dataset['test']
        logging.info(f"Dataset loaded with {len(test_dataset)} test samples")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return
    
    # Create output directory
    test_output_dir = prepare_output_dir(output_dir, model_name, 'marathi-asr')
    
    # Initialize CSV file
    csv_path = os.path.join(test_output_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvfile.write('file_id,ground_truth,hypothesis,wer,cer\n')
    
    # Process samples in temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        logging.debug(f"Created temporary directory: {tmpdir}")
        
        # Initialize metrics
        total_duration = 0.0
        results = []
        processed_count = 0
        error_count = 0
        max_errors = 5
        
        # Process each sample
        for i, sample in enumerate(tqdm(test_dataset, desc="Processing Marathi samples")):
            if total_duration >= OUTPUT_CONFIG['max_audio_duration']:
                logging.info(f"Reached maximum audio duration limit ({OUTPUT_CONFIG['max_audio_duration']}s)")
                break
            
            # Generate file ID
            file_id = f"sample_{i:04d}"
            
            try:
                # Extract audio data
                audio_array = sample['audio']['array']
                sampling_rate = sample['audio']['sampling_rate']
                
                # Save audio to temporary file
                local_audio_path = os.path.join(tmpdir, f"{file_id}.wav")
                sf.write(local_audio_path, audio_array, sampling_rate)
                
                # Get audio duration
                duration = get_audio_duration(local_audio_path)
                total_duration += duration
                
                # Get ground truth
                ground_truth = sample['transcriptions']
                
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
                
                processed_count += 1
                
            except Exception as e:
                logging.error(f"Error processing {file_id}: {e}")
                error_count += 1
                if error_count >= max_errors:
                    logging.error(f"Too many errors ({error_count}), stopping")
                    break
        
        # Calculate metrics
        if not results:
            logging.warning("No results collected")
            return
        
        metrics = calculate_metrics(results)
        total_hours = total_duration / 3600.0
        
        # Save metrics
        metrics_path = os.path.join(test_output_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': 'marathi-asr',
                'model': model_name,
                'num_samples_processed': processed_count,
                'total_samples_in_dataset': len(test_dataset),
                'total_duration_seconds': total_duration,
                'total_duration_hours': total_hours,
                'avg_wer': metrics['avg_wer'],
                'avg_cer': metrics['avg_cer']
            }, f, indent=2)
        
        # Log results
        logging.info("\n=== Marathi ASR Results ===")
        logging.info(f"Model: {model_name}")
        logging.info(f"Samples processed: {processed_count}/{len(test_dataset)}")
        logging.info(f"Total duration: {total_duration:.2f}s ({total_hours:.2f}h)")
        logging.info(f"Average WER: {metrics['avg_wer']:.4f}")
        logging.info(f"Average CER: {metrics['avg_cer']:.4f}")
        
        # Save overall summary
        overall_metrics_path = os.path.join(output_dir, f"{model_name}_marathi_asr_metrics.json")
        with open(overall_metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': model_name,
                'dataset': 'TheAIchemist13/marathi_asr_dataset',
                'results': {
                    'num_samples_processed': processed_count,
                    'total_samples_in_dataset': len(test_dataset),
                    'total_duration_seconds': total_duration,
                    'total_duration_hours': total_hours,
                    'avg_wer': metrics['avg_wer'],
                    'avg_cer': metrics['avg_cer']
                }
            }, f, indent=2)


# ==============================================================================
# Flexible dataset entry points for Marathi
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


def _marathi_iter_hf_samples(dataset_name: str, split: str, tmpdir: str, audio_col: str = 'audio', text_col: str = 'text'):
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
        try:
            if isinstance(audio, dict) and 'array' in audio and 'sampling_rate' in audio:
                local_path = os.path.join(tmpdir, f"{idx}.wav")
                sf.write(local_path, audio['array'], audio['sampling_rate'])
            elif hasattr(audio, 'array') and hasattr(audio, 'sampling_rate'):
                local_path = os.path.join(tmpdir, f"{idx}.wav")
                sf.write(local_path, audio.array, audio.sampling_rate)
            elif isinstance(audio, str):
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

def _marathi_iter_csv_samples(csv_path: str, tmpdir: str, default_bucket: str):
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

def process_marathi(
    model_name: str,
    model_config: Dict[str, Any],
    dataset_key: str = 'marathi-asr',
    output_dir: Optional[str] = None,
    hf_dataset_name: Optional[str] = None,
    hf_split: str = 'test',
    csv_path: Optional[str] = None,
) -> None:
    """
    Unified Marathi entrypoint. Supports:
      • HuggingFace dataset (default: TheAIchemist13/marathi_asr_dataset)
      • A custom CSV with columns: Key|audio_path, ground_truth|Transcription
    """
    if output_dir is None:
        output_dir = OUTPUT_CONFIG['base_dir']

    logging.info(f"Initializing {model_name} model...")
    model = get_model(model_name, model_config)
    model.load()

    cfg = DATASET_CONFIGS.get(dataset_key, {})
    source = cfg.get('source') or ('huggingface' if hf_dataset_name else 'csv' if csv_path else 'huggingface')

    if source == 'csv' and csv_path:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        test_output_dir = prepare_output_dir(output_dir, model_name, dataset_name)
        results_csv = os.path.join(test_output_dir, 'results.csv')
        os.makedirs(test_output_dir, exist_ok=True)
        with open(results_csv, 'w', newline='', encoding='utf-8') as csvfile:
            csvfile.write('file_id,ground_truth,hypothesis,wer,cer\n')
        total_duration = 0.0
        results = []
        with tempfile.TemporaryDirectory() as tmpdir:
            default_bucket = S3_CONFIG.get('bucket_name') if 'S3_CONFIG' in globals() else None
            for file_id, local_path, ground_truth in _marathi_iter_csv_samples(csv_path, tmpdir, default_bucket):
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
        if results:
            metrics = calculate_metrics(results)
            with open(os.path.join(test_output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
                json.dump({'dataset': dataset_name, 'model': model_name, 'avg_wer': metrics['avg_wer'], 'avg_cer': metrics['avg_cer'], 'num_files': metrics['num_files']}, f, indent=2)
        return

    # HuggingFace default
    dataset_name = hf_dataset_name or cfg.get('dataset_name') or 'TheAIchemist13/marathi_asr_dataset'
    dataset_tag = dataset_name.split('/')[-1]
    test_output_dir = prepare_output_dir(output_dir, model_name, f"hf-{dataset_tag}")
    results_csv = os.path.join(test_output_dir, 'results.csv')
    os.makedirs(test_output_dir, exist_ok=True)
    with open(results_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvfile.write('file_id,ground_truth,hypothesis,wer,cer\n')
    total_duration = 0.0
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for file_id, local_path, ground_truth in _marathi_iter_hf_samples(dataset_name, hf_split, tmpdir):
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
    if results:
        metrics = calculate_metrics(results)
        with open(os.path.join(test_output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump({'dataset': dataset_tag, 'model': model_name, 'avg_wer': metrics['avg_wer'], 'avg_cer': metrics['avg_cer'], 'num_files': metrics['num_files']}, f, indent=2)
    return
