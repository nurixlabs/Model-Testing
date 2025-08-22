#!/usr/bin/env python3
"""
Main entry point for the Speech-to-Text evaluation pipeline
Unified interface for English, Marathi, and Hinglish/Hindi pipelines.

All pipelines now support:
- Default dataset per language (EN: LibriSpeech; MR: HF dataset; HI/Hinglish: CSV)
- Hugging Face datasets via --hf-dataset-name/--hf-split
- CSV via --csv with columns: Key|audio_path and Transcription|ground_truth

Keep the pipelines' internal logic as-is; this file only routes arguments.
"""
import os
import sys
import argparse
import logging
from typing import Optional, Dict, Any
# Updated pipeline imports (each exposes a single entrypoint)
from pipelines.english_pipeline_updated import process_english
from pipelines.marathi_pipeline_updated import process_marathi
from pipelines.hinglish_pipeline_updated import process_hinglish

# Import configuration
from config import (
    AVAILABLE_MODELS,
    LANGUAGE_CONFIGS,
    DATASET_CONFIGS,
    MODEL_CONFIGS,
    API_KEYS,
    OUTPUT_CONFIG
)

# Set up logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO
)


def validate_model_for_language(model: str, language: str) -> bool:
    """
    Validate if a model supports the specified language.
    """
    if language not in LANGUAGE_CONFIGS:
        return True  # Allow any model for unknown languages
    supported_models = LANGUAGE_CONFIGS[language].get('models', AVAILABLE_MODELS)
    return model in supported_models


def get_model_config(model: str, language: str) -> Dict[str, Any]:
    """
    Get model configuration with language-specific overrides and API keys.
    """
    config = MODEL_CONFIGS.get(model, {}).copy()

    # Apply language-specific overrides
    if language in LANGUAGE_CONFIGS:
        default_lang_code = LANGUAGE_CONFIGS[language].get('default_language_code')
        if default_lang_code:
            # Update language codes based on model type
            if 'language_code' in config:
                config['language_code'] = default_lang_code
            elif 'language' in config:
                # Keep full locale for providers that require it (e.g., Azure)
                if model == 'azure':
                    config['language'] = default_lang_code
                else:
                    # Extract base language (e.g., 'hi' from 'hi-IN')
                    config['language'] = default_lang_code.split('-')[0]

    # Add API key if available
    if model in API_KEYS and API_KEYS[model]:
        config['api_key'] = API_KEYS[model]

    return config


def main():
    parser = argparse.ArgumentParser(
        description='Speech-to-Text Evaluation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ENGLISH — default LibriSpeech (S3) with Whisper
  %(prog)s --dataset librispeech --model whisper --test-set test-clean

  # ENGLISH — HF dataset override
  %(prog)s --dataset librispeech --model whisper --hf-dataset-name mozilla-foundation/common_voice_16_1 --hf-split test

  # ENGLISH — CSV override
  %(prog)s --dataset librispeech --model whisper --csv /path/to/english.csv

  # MARATHI — default HF dataset with Sarvam
  %(prog)s --dataset marathi-asr --model sarvam

  # MARATHI — CSV override
  %(prog)s --dataset marathi-asr --model sarvam --csv /path/to/marathi.csv

  # HINGLISH/HINDI — default CSV (must provide --csv)
  %(prog)s --dataset custom-csv --model deepgram --csv /path/to/hinglish.csv --language hinglish

  # HINGLISH/HINDI — use HF dataset instead of CSV
  %(prog)s --dataset custom-csv --model whisper --language hinglish --hf-dataset-name ai4bharat/hinglish_asr --hf-split test
        """
    )
    parser.add_argument('--model', required=True,
                    choices=AVAILABLE_MODELS,
                    help='Model to use for transcription')

    # Required arguments
    parser.add_argument('--dataset',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset key to process (selects defaults per language)')

    # Optional arguments
    parser.add_argument('--language',
                        choices=list(LANGUAGE_CONFIGS.keys()),
                        help='Override language (default: dataset-specific)')
    parser.add_argument('--test-set',
                        choices=['test-clean', 'test-other', 'both'],
                        default='both',
                        help='Test set for LibriSpeech (default: both)')
    parser.add_argument('--csv',
                        help='Path to CSV file (columns: Key|audio_path and Transcription|ground_truth)')
    parser.add_argument('--hf-dataset-name',
                        help='Hugging Face dataset name, e.g., mozilla-foundation/common_voice_16_1')
    parser.add_argument('--hf-split',
                        default='test',
                        help='Hugging Face split to use (default: test)')
    parser.add_argument('--output-dir',
                        default=OUTPUT_CONFIG['base_dir'],
                        help=f'Output directory (default: {OUTPUT_CONFIG["base_dir"]})')
    parser.add_argument('--gcloud-path',
                        default='./google-cloud-sdk/bin/gcloud',
                        help='Path to gcloud executable (for Google models)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine language (CLI override > dataset default > english)
    dataset_config = DATASET_CONFIGS[args.dataset]
    language = args.language or dataset_config.get('language', 'english')

    # Validate model for language
    if not validate_model_for_language(args.model, language):
        supported_models = LANGUAGE_CONFIGS[language].get('models', [])
        logging.error(f"Model '{args.model}' does not support language '{language}'")
        if supported_models:
            logging.info(f"Supported models for {language}: {', '.join(supported_models)}")
        sys.exit(1)

    # Get model configuration
    model_config = get_model_config(args.model, language)

    # Add gcloud path for Google model
    if args.model in ['google', 'google_v2']:
        model_config['gcloud_path'] = args.gcloud_path

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Infer dataset "mode" for logging/validation
    dataset_mode = 'default'
    hf_name = getattr(args, 'hf_dataset_name', None)
    if args.csv:
        dataset_mode = 'csv'
    if hf_name:
        dataset_mode = 'huggingface'

    logging.info(f"Dataset key: {args.dataset} (mode: {dataset_mode})")
    logging.info(f"Language: {language}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Output directory: {args.output_dir}")

    # Basic validation
    if language in ('hinglish', 'hindi') and not (args.csv or hf_name):
        logging.error("Hinglish/Hindi requires either --csv or --hf-dataset-name (default is CSV).")
        sys.exit(1)

    if dataset_mode == 'csv' and args.csv and not os.path.exists(args.csv) and not args.csv.startswith('s3://'):
        logging.error(f"CSV file not found: {args.csv}")
        sys.exit(1)

    # Route to appropriate pipeline
    try:
        if language == 'english':
            # process_english supports: default LibriSpeech OR HF OR CSV
            process_english(
                model_name=args.model,
                model_config=model_config,
                output_dir=args.output_dir,
                # defaults (used only if neither csv nor HF provided)
                dataset_key=args.dataset,
                test_set=args.test_set,
                # overrides
                csv_path=args.csv,
                hf_dataset_name=hf_name,
                hf_split=args.hf_split
            )

        elif language == 'marathi':
            # process_marathi supports: default HF OR CSV (or alternative HF)
            process_marathi(
                model_name=args.model,
                model_config=model_config,
                output_dir=args.output_dir,
                dataset_key=args.dataset,
                csv_path=args.csv,
                hf_dataset_name=hf_name,
                hf_split=args.hf_split
            )

        elif language in ('hinglish', 'hindi'):
            # process_hinglish supports: CSV (default) OR HF
            process_hinglish(
                model_name=args.model,
                model_config=model_config,
                output_dir=args.output_dir,
                dataset_key=args.dataset,
                csv_path=args.csv,
                hf_dataset_name=hf_name,
                hf_split=args.hf_split,
                language=language
            )

        else:
            logging.error(f"Unknown or unsupported language: {language}")
            sys.exit(1)

    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=args.debug)
        sys.exit(1)


if __name__ == "__main__":
    main()