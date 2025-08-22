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