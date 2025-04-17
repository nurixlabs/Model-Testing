#!/usr/bin/env python3
"""
Evaluate transcription results by calculating WER and CER metrics.
This script can be used independently to evaluate any set of transcription results.
"""

import os
import json
import csv
import argparse
import glob
from jiwer import wer, cer
import pandas as pd

def load_reference_transcripts(reference_path):
    """
    Load reference transcripts from a file or directory.
    
    Parameters:
        reference_path (str): Path to a .trans.txt file or directory containing .trans.txt files
        
    Returns:
        dict: Mapping of file ids to reference transcripts
    """
    transcript_dict = {}
    
    if os.path.isdir(reference_path):
        # Find all .trans.txt files in the directory
        trans_files = glob.glob(os.path.join(reference_path, "**/*.trans.txt"), recursive=True)
        for trans_file in trans_files:
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        key, transcript = parts
                        transcript_dict[key] = transcript.lower()
    else:
        # Load a single transcript file
        with open(reference_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    key, transcript = parts
                    transcript_dict[key] = transcript.lower()
    
    return transcript_dict

def load_hypothesis_transcripts(hypothesis_path):
    """
    Load hypothesis transcripts from JSON files.
    
    Parameters:
        hypothesis_path (str): Path to directory containing JSON files
        
    Returns:
        dict: Mapping of file ids to hypothesis transcripts
    """
    hypothesis_dict = {}
    
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(hypothesis_path, "*.json"))
    
    for json_file in json_files:
        file_id = os.path.splitext(os.path.basename(json_file))[0]
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # Extract the transcript text
                if 'text' in data:
                    hypothesis_dict[file_id] = data['text'].lower()
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON file {json_file}")
    
    return hypothesis_dict

def calculate_metrics(references, hypotheses):
    """
    Calculate WER and CER metrics for each file.
    
    Parameters:
        references (dict): Mapping of file ids to reference transcripts
        hypotheses (dict): Mapping of file ids to hypothesis transcripts
        
    Returns:
        list: List of dictionaries with file_id, reference, hypothesis, wer, and cer
    """
    results = []
    
    for file_id, reference in references.items():
        if file_id in hypotheses:
            hypothesis = hypotheses[file_id]
            file_wer = wer(reference, hypothesis)
            file_cer = cer(reference, hypothesis)
            
            results.append({
                'file_id': file_id,
                'reference': reference,
                'hypothesis': hypothesis,
                'wer': file_wer,
                'cer': file_cer
            })
    
    return results

def save_results_to_csv(results, output_csv):
    """
    Save evaluation results to CSV.
    
    Parameters:
        results (list): List of result dictionaries
        output_csv (str): Path to output CSV file
    """
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_id', 'reference', 'hypothesis', 'wer', 'cer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def save_summary(results, output_json):
    """
    Save summary metrics to JSON.
    
    Parameters:
        results (list): List of result dictionaries
        output_json (str): Path to output JSON file
    """
    if not results:
        summary = {'avg_wer': 0, 'avg_cer': 0, 'num_files': 0}
    else:
        avg_wer = sum(r['wer'] for r in results) / len(results)
        avg_cer = sum(r['cer'] for r in results) / len(results)
        summary = {
            'avg_wer': avg_wer,
            'avg_cer': avg_cer,
            'num_files': len(results)
        }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Evaluate speech transcription results')
    parser.add_argument('--reference', required=True, help='Path to reference transcripts (.trans.txt file or directory)')
    parser.add_argument('--hypothesis', required=True, help='Path to hypothesis JSON files directory')
    parser.add_argument('--output-csv', default='evaluation_results.csv', help='Path to output CSV file')
    parser.add_argument('--output-json', default='evaluation_summary.json', help='Path to output JSON summary file')
    
    args = parser.parse_args()
    
    # Load reference and hypothesis transcripts
    print(f"Loading reference transcripts from {args.reference}...")
    references = load_reference_transcripts(args.reference)
    print(f"Found {len(references)} reference transcripts.")
    
    print(f"Loading hypothesis transcripts from {args.hypothesis}...")
    hypotheses = load_hypothesis_transcripts(args.hypothesis)
    print(f"Found {len(hypotheses)} hypothesis transcripts.")
    
    # Calculate metrics
    print("Calculating metrics...")
    results = calculate_metrics(references, hypotheses)
    
    # Save results
    print(f"Saving detailed results to {args.output_csv}...")
    save_results_to_csv(results, args.output_csv)
    
    print(f"Saving summary to {args.output_json}...")
    save_summary(results, args.output_json)
    
    # Print summary
    if results:
        avg_wer = sum(r['wer'] for r in results) / len(results)
        avg_cer = sum(r['cer'] for r in results) / len(results)
        print("\nEvaluation Summary:")
        print(f"Files evaluated: {len(results)}")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Average CER: {avg_cer:.4f}")
    else:
        print("\nNo matching files found for evaluation.")

if __name__ == '__main__':
    main()