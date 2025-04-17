#!/usr/bin/env python3
"""
Speech-to-Text Transcription Pipeline

This script orchestrates the transcription process for speech files using
various STT engines (Dolphin, Whisper, Google, AWS, Salad, Deepgram).
It handles downloading audio from S3, transcription, and evaluation.
"""

import os
import argparse
import tempfile
import json
from tqdm import tqdm

from config import S3_CONFIG, MODEL_CONFIGS, OUTPUT_CONFIG, AVAILABLE_MODELS, API_KEYS
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

def process_dataset(model, bucket_name, prefix, output_dir, test_set):
    """
    Process an entire dataset using the specified model.
    
    Parameters:
        model: An instance of a model class
        bucket_name (str): S3 bucket name
        prefix (str): S3 prefix for dataset
        output_dir (str): Base output directory
        test_set (str): Test set name (e.g., 'test-clean')
        
    Returns:
        dict: Metrics for this dataset
    """
    print(f"\nProcessing dataset: {test_set} from s3://{bucket_name}/{prefix}")
    
    # Create output directory for this test set
    test_output_dir = prepare_output_dir(output_dir, model.name, test_set)
    
    # List audio and transcript files
    audio_files = list_files_in_s3(bucket_name, prefix, ('.wav', '.mp3', '.flac'))
    transcript_files = list_files_in_s3(bucket_name, prefix, ('.trans.txt',))
    
    if not audio_files:
        print(f"No audio files found in s3://{bucket_name}/{prefix}. Skipping.")
        return None
    
    if not transcript_files:
        print(f"No transcript files found in s3://{bucket_name}/{prefix}. Skipping.")
        return None
    
    print(f"Found {len(audio_files)} audio file(s) and {len(transcript_files)} transcript file(s).")
    
    # Initialize CSV file with header
    csv_path = os.path.join(test_output_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvfile.write('file_id,ground_truth,hypothesis,wer,cer\n')
    
    # Use a temporary directory for downloaded files
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Created temporary directory '{tmpdir}' for downloading files.")
        
        # Build the transcript dictionary
        print("Building transcript dictionary...")
        transcript_dict = build_transcript_dict(bucket_name, transcript_files, tmpdir)
        
        # Initialize metrics
        total_duration = 0.0
        results = []
        
        # Process each audio file
        for audio_file_key in tqdm(audio_files, desc=f"Processing {test_set}"):
            if total_duration >= OUTPUT_CONFIG['max_audio_duration']:
                print(f"Reached maximum audio duration ({OUTPUT_CONFIG['max_audio_duration']} seconds). Stopping.")
                break
            
            # Get file ID (basename without extension)
            file_id = os.path.splitext(os.path.basename(audio_file_key))[0]
            
            # Skip if we don't have a ground truth for this file
            if file_id not in transcript_dict:
                print(f"No ground truth found for {file_id}, skipping.")
                continue
            
            try:
                # Download the audio file
                local_audio_path = download_file_from_s3(bucket_name, audio_file_key, tmpdir)
                
                # Get audio duration
                duration = get_audio_duration(local_audio_path)
                total_duration += duration
                
                # Get ground truth
                ground_truth = transcript_dict[file_id]
                
                # Transcribe using the model
                print(f"Transcribing {file_id}...")
                result = model.transcribe(local_audio_path)
                
                # Save result with ground truth
                save_result(test_output_dir, file_id, result, ground_truth)
                
                # Track results for metrics calculation
                if 'wer' in result and 'cer' in result:
                    results.append({
                        'file_id': file_id,
                        'wer': result['wer'],
                        'cer': result['cer']
                    })
                
            except Exception as e:
                print(f"Error processing {file_id}: {e}")
        
        # Calculate and save metrics
        metrics = calculate_metrics(results)
        
        # Save metrics to JSON
        metrics_path = os.path.join(test_output_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'test_set': test_set,
                'model': model.name,
                'num_files': metrics['num_files'],
                'total_duration': total_duration,
                'avg_wer': metrics['avg_wer'],
                'avg_cer': metrics['avg_cer']
            }, f, indent=2)
        
        # Print metrics
        print(f"\n{test_set} Results:")
        print(f"Number of files processed: {metrics['num_files']}")
        print(f"Total audio duration: {total_duration:.2f} seconds")
        print(f"Average WER: {metrics['avg_wer']:.4f}")
        print(f"Average CER: {metrics['avg_cer']:.4f}")
        
        return {
            'num_files': metrics['num_files'],
            'total_duration': total_duration,
            'avg_wer': metrics['avg_wer'],
            'avg_cer': metrics['avg_cer']
        }

def main():
    parser = argparse.ArgumentParser(description='Speech-to-Text Transcription Pipeline')
    parser.add_argument('--model', required=True, choices=AVAILABLE_MODELS,
                      help='Model to use for transcription')
    parser.add_argument('--test-set', choices=['test-clean', 'test-other', 'both'], default='both',
                      help='Test set to process (default: both)')
    parser.add_argument('--output-dir', default=None,
                      help=f'Output directory (default: {OUTPUT_CONFIG["base_dir"]})')
    parser.add_argument('--api-key', default=None,
                      help='API key for cloud services (if not set in environment/config)')
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir or OUTPUT_CONFIG['base_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Update API key if provided
    if args.api_key:
        API_KEYS[args.model] = args.api_key
    
    # Load model configuration
    model_config = MODEL_CONFIGS.get(args.model, {})
    
    # Add API key to the config if available
    if args.model in API_KEYS and API_KEYS[args.model]:
        model_config['api_key'] = API_KEYS[args.model]
    
    # Create and initialize the model
    try:
        print(f"Initializing {args.model} model...")
        model = get_model(args.model, model_config)
        model.load()
    except Exception as e:
        print(f"Error initializing {args.model} model: {e}")
        return
    
    # Process the specified test sets
    results = {}
    
    if args.test_set in ['test-clean', 'both']:
        print("\nProcessing test-clean dataset...")
        test_clean_metrics = process_dataset(
            model, 
            S3_CONFIG['bucket_name'], 
            S3_CONFIG['test_clean_prefix'], 
            output_dir, 
            'test-clean'
        )
        if test_clean_metrics:
            results['test-clean'] = test_clean_metrics
    
    if args.test_set in ['test-other', 'both']:
        print("\nProcessing test-other dataset...")
        test_other_metrics = process_dataset(
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
        overall_metrics_path = os.path.join(output_dir, f"{args.model}_overall_metrics.json")
        with open(overall_metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': args.model,
                'results': results
            }, f, indent=2)
        
        print("\n=== OVERALL RESULTS ===")
        for test_set, metrics in results.items():
            print(f"{test_set}: WER={metrics['avg_wer']:.4f}, CER={metrics['avg_cer']:.4f}, Files={metrics['num_files']}")
    
    print(f"\nAll results saved to {output_dir}")

if __name__ == "__main__":
    main()