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