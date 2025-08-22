#!/usr/bin/env python3
"""
Hinglish/Code-Mixed Dataset Processing Pipeline
Handles custom CSV files with LLM-based evaluation
"""
import os
import csv
import tempfile
import json
import logging
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from openai import OpenAI

from config import S3_CONFIG
from utils import (
    download_file_from_s3,
    prepare_output_dir
)
from models.model_factory import get_model


# Initialize OpenAI client (loaded from environment)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Parse an S3 URI and return (bucket_name, key).
    
    Args:
        s3_uri: Full S3 URI like 's3://bucket-name/path/to/file.mp3'
    
    Returns:
        tuple: (bucket_name, key_path)
    """
    if not s3_uri.startswith('s3://'):
        # If it's already just a key path, return None for bucket
        return None, s3_uri

    # Remove 's3://' prefix
    path = s3_uri[5:]

    # Split on first '/' to separate bucket from key
    parts = path.split('/', 1)
    if len(parts) == 2:
        bucket_name, key = parts
        return bucket_name, key
    else:
        # No key path, just bucket
        return parts[0], ''


def llm_judge_score(
    ground_truth: str, 
    hypothesis: str, 
    model_name: str = "gpt-4o"
) -> Optional[int]:
    """
    Use an LLM to rate the transcription quality on a 1-5 semantic scale.
    
    Args:
        ground_truth: Reference transcription
        hypothesis: Model's transcription output
        model_name: OpenAI model to use for evaluation
        
    Returns:
        int: Score from 1-5, or None if evaluation fails
    """
    prompt = (
        "TASK: Compare the ground truth transcript with the STT hypothesis and rate semantic similarity.\n"
        "The hypothesis may be in Roman script, Devanagari script, or a mixture of both.\n\n"
        "SCORING SCALE (1-5):\n"
        "5 = EXCELLENT: Perfect semantic match. Meaning, context, and code-switching patterns are preserved.\n"
        "4 = GOOD: Minor differences in script or word choice, but core meaning intact.\n"
        "3 = FAIR: Semantic meaning mostly preserved with some loss of nuance.\n"
        "2 = POOR: Partial semantic preservation; some key information lost.\n"
        "1 = VERY POOR: Major semantic differences; core meaning lost or unintelligible.\n\n"
        "EVALUATION CRITERIA:\n"
        "• Semantic meaning preservation (40%)\n"
        "• Key information accuracy (25%)\n"
        "• Code-switching naturalness (20%)\n"
        "• Script handling appropriateness (15%)\n\n"
        "IMPORTANT: Focus on semantic similarity, not exact word matching.\n\n"
        "Return only the numeric score (1, 2, 3, 4, or 5).\n\n"
        f"Ground Truth: \"{ground_truth}\"\n"
        f"Hypothesis:   \"{hypothesis}\""
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert evaluator for Hinglish (Hindi-English code-mixed) ASR output."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        # Extract numeric score
        score = int(''.join(filter(str.isdigit, content)))
        if 1 <= score <= 5:
            return score
        else:
            logging.warning(f"LLM returned out-of-range score: {score}")
            return None
            
    except Exception as e:
        logging.error(f"LLM evaluation failed: {e}")
        return None


def process_csv_batch(
    rows: List[Dict[str, str]],
    model: Any,
    bucket_name: str,
    tmpdir: str,
    writer: csv.writer
) -> List[Dict[str, Any]]:
    """
    Process a batch of CSV rows.
    
    Args:
        rows: List of CSV row dictionaries
        model: Loaded STT model instance
        bucket_name: Default S3 bucket name
        tmpdir: Temporary directory for downloads
        writer: CSV writer object
        
    Returns:
        list: Results with scores
    """
    results = []
    
    for row in rows:
        # Extract S3 path
        s3_path = row.get('Key') or row.get('audio_path')
        if not s3_path:
            logging.warning(f"Skipping row with no audio path: {row}")
            continue
        
        # Parse S3 URI
        parsed_bucket, s3_key = parse_s3_uri(s3_path)
        target_bucket = parsed_bucket if parsed_bucket else bucket_name
        
        # Generate file ID
        file_id = os.path.splitext(os.path.basename(s3_key))[0]
        
        # Get ground truth
        ground_truth = row.get('ground_truth') or row.get('Transcription')
        if not ground_truth:
            logging.warning(f"Skipping row with no ground truth: {row}")
            continue
        
        try:
            # Download audio file
            local_path = download_file_from_s3(target_bucket, s3_key, tmpdir)
            
            # Transcribe
            logging.info(f"Transcribing {file_id}...")
            result = model.transcribe(local_path)
            hypothesis = result.get('transcript') or result.get('text', '')
            
            # Evaluate with LLM
            score = llm_judge_score(ground_truth, hypothesis)
            
            # Write result
            writer.writerow([file_id, ground_truth, hypothesis, score])
            
            results.append({
                'file_id': file_id,
                'score': score
            })
            
        except Exception as e:
            if any(code in str(e) for code in ("404", "Not Found", "NoSuchKey")):
                logging.warning(f"Audio not found: {s3_path}")
            else:
                logging.error(f"Error processing {file_id}: {e}")
    
    return results


def process_hinglish_csv(
    model_name: str,
    model_config: Dict[str, Any],
    csv_path: str,
    output_dir: str = None,
    language: str = 'hinglish'
) -> None:
    """
    Process a CSV file containing Hinglish/code-mixed audio files.
    
    Args:
        model_name: Name of the model to use
        model_config: Model configuration dictionary
        csv_path: Path to input CSV file
        output_dir: Output directory path
        language: Language code for the dataset
    """
    # Set default output directory
    if output_dir is None:
        output_dir = OUTPUT_CONFIG['base_dir']
    
    # Create and load model
    logging.info(f"Initializing {model_name} model...")
    model = get_model(model_name, model_config)
    model.load()
    
    # Prepare output directory
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    test_output_dir = prepare_output_dir(output_dir, model_name, dataset_name)
    
    # Initialize results CSV
    results_csv = os.path.join(test_output_dir, 'results.csv')
    with open(results_csv, 'w', newline='', encoding='utf-8') as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(['file_id', 'ground_truth', 'hypothesis', 'llm_score'])
        
        # Process CSV in temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            logging.debug(f"Created temporary directory: {tmpdir}")
            
            # Read and process CSV
            all_results = []
            batch_size = 10  # Process in batches for better progress tracking
            
            with open(csv_path, encoding='utf-8') as in_csv:
                reader = list(csv.DictReader(in_csv))
                total_rows = len(reader)
                
                logging.info(f"Processing {total_rows} audio files from CSV")
                
                # Process in batches
                for i in range(0, total_rows, batch_size):
                    batch = reader[i:i+batch_size]
                    batch_results = process_csv_batch(
                        batch, 
                        model, 
                        S3_CONFIG['bucket_name'],
                        tmpdir,
                        writer
                    )
                    all_results.extend(batch_results)
                    
                    # Progress update
                    processed = min(i + batch_size, total_rows)
                    logging.info(f"Processed {processed}/{total_rows} files")
    
    # Calculate metrics
    valid_scores = [r['score'] for r in all_results if r['score'] is not None]
    
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        score_distribution = {i: valid_scores.count(i) for i in range(1, 6)}
        
        # Save metrics
        metrics_path = os.path.join(test_output_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset_name,
                'model': model_name,
                'language': language,
                'total_files': len(all_results),
                'files_evaluated': len(valid_scores),
                'average_score': avg_score,
                'score_distribution': score_distribution
            }, f, indent=2)
        
        # Log results
        logging.info(f"\n=== {dataset_name} Results ===")
        logging.info(f"Model: {model_name}")
        logging.info(f"Language: {language}")
        logging.info(f"Files processed: {len(all_results)}")
        logging.info(f"Files evaluated: {len(valid_scores)}")
        logging.info(f"Average LLM Score: {avg_score:.2f}/5.0")
        logging.info("Score Distribution:")
        for score, count in sorted(score_distribution.items()):
            logging.info(f"  Score {score}: {count} files ({count/len(valid_scores)*100:.1f}%)")
    else:
        logging.warning("No valid scores collected")
    
    logging.info(f"Results saved to {results_csv}")