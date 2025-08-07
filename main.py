#!/usr/bin/env python3
"""
Main entry point for the Speech-to-Text evaluation pipeline
Handles different languages and datasets through a unified interface
"""
import os
import sys
import argparse
import logging
from typing import Optional, Dict, Any

# Import pipeline modules
from pipelines.english_pipeline import process_librispeech
from pipelines.marathi_pipeline import process_marathi_asr
from pipelines.hinglish_pipeline import process_hinglish_csv

# Import configuration
from config import (
    AVAILABLE_MODELS, 
    LANGUAGE_CONFIGS, 
    DATASET_CONFIGS,
    MODEL_CONFIGS,
    API_KEYS,
    OUTPUT_CONFIG,
    LANGUAGE_MAPPING,
    MODEL_LANGUAGE_SUPPORT,
    QUICK_REFERENCE
)

# Set up logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO
)


def validate_model_for_language(model: str, language: str) -> bool:
    """
    Validate if a model supports the specified language.
    
    Args:
        model: Model name
        language: Language code
        
    Returns:
        bool: True if model supports the language
    """
    if language not in LANGUAGE_CONFIGS:
        return True  # Allow any model for unknown languages
    
    supported_models = LANGUAGE_CONFIGS[language].get('models', AVAILABLE_MODELS)
    return model in supported_models


def get_model_config(model: str, language: str) -> Dict[str, Any]:
    """
    Get model configuration with language-specific overrides.
    
    Args:
        model: Model name
        language: Language code
        
    Returns:
        dict: Model configuration
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
                # Extract language part from language code (e.g., 'hi' from 'hi-IN')
                config['language'] = default_lang_code.split('-')[0]
    
    # Add API key if available
    if model in API_KEYS and API_KEYS[model]:
        config['api_key'] = API_KEYS[model]
    
    return config


def find_models_for_language(language: str) -> Dict[str, Any]:
    """
    Find all models that support a given language.
    
    Args:
        language: Language name or code (e.g., 'english', 'hi-IN', 'en-US')
        
    Returns:
        dict: Information about supported models
    """
    supported_models = []
    language_info = {}
    
    # Check if it's a language code
    for lang_key, lang_data in LANGUAGE_MAPPING.items():
        if language.lower() in [lang_key.lower()] + [code.lower() for code in lang_data['language_codes']]:
            language_info = lang_data
            supported_models = lang_data['models']
            break
    
    if not supported_models:
        return {
            'error': f'Language "{language}" not found. Available languages: {list(LANGUAGE_MAPPING.keys())}'
        }
    
    # Get detailed model information
    model_details = {}
    for model in supported_models:
        if model in MODEL_LANGUAGE_SUPPORT:
            model_details[model] = MODEL_LANGUAGE_SUPPORT[model]
    
    return {
        'language': language_info,
        'supported_models': supported_models,
        'model_details': model_details,
        'recommendation': language_info.get('best_model', supported_models[0]) if supported_models else None
    }


def find_languages_for_model(model: str) -> Dict[str, Any]:
    """
    Find all languages supported by a given model.
    
    Args:
        model: Model name
        
    Returns:
        dict: Information about supported languages
    """
    if model not in MODEL_LANGUAGE_SUPPORT:
        return {
            'error': f'Model "{model}" not found. Available models: {list(MODEL_LANGUAGE_SUPPORT.keys())}'
        }
    
    model_info = MODEL_LANGUAGE_SUPPORT[model]
    supported_languages = []
    
    for lang_key, lang_data in LANGUAGE_MAPPING.items():
        if model in lang_data['models']:
            supported_languages.append({
                'language': lang_data['display_name'],
                'language_codes': lang_data['language_codes'],
                'description': lang_data['description']
            })
    
    return {
        'model': model_info,
        'supported_languages': supported_languages
    }


def get_quick_recommendations(use_case: str) -> Dict[str, Any]:
    """
    Get quick model recommendations for common use cases.
    
    Args:
        use_case: Use case (e.g., 'english_speech', 'hindi_speech', 'offline_processing')
        
    Returns:
        dict: Recommendations for the use case
    """
    if use_case not in QUICK_REFERENCE:
        available_cases = list(QUICK_REFERENCE.keys())
        return {
            'error': f'Use case "{use_case}" not found. Available use cases: {available_cases}'
        }
    
    return QUICK_REFERENCE[use_case]


def main():
    parser = argparse.ArgumentParser(
        description='Speech-to-Text Evaluation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process LibriSpeech test-clean with Whisper
  %(prog)s --dataset librispeech --model whisper --test-set test-clean
  
  # Process Marathi ASR dataset with Sarvam
  %(prog)s --dataset marathi-asr --model sarvam
  
  # Process custom CSV with Hinglish content
  %(prog)s --dataset custom-csv --model deepgram --csv path/to/file.csv --language hinglish
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', required=True, 
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to process')
    parser.add_argument('--model', required=True, 
                        choices=AVAILABLE_MODELS,
                        help='Model to use for transcription')
    
    # Optional arguments
    parser.add_argument('--language', 
                        choices=list(LANGUAGE_CONFIGS.keys()),
                        help='Override language (default: dataset-specific)')
    parser.add_argument('--test-set', 
                        choices=['test-clean', 'test-other', 'both'],
                        default='both',
                        help='Test set for LibriSpeech (default: both)')
    parser.add_argument('--csv', 
                        help='Path to CSV file (required for custom-csv dataset)')
    parser.add_argument('--output-dir', 
                        default=OUTPUT_CONFIG['base_dir'],
                        help=f'Output directory (default: {OUTPUT_CONFIG["base_dir"]})')
    parser.add_argument('--gcloud-path', 
                        default='./google-cloud-sdk/bin/gcloud',
                        help='Path to gcloud executable')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    # Language mapping utility arguments
    parser.add_argument('--find-models', 
                        help='Find models that support a specific language (e.g., "english", "hi-IN")')
    parser.add_argument('--find-languages', 
                        help='Find languages supported by a specific model (e.g., "whisper", "sarvam")')
    parser.add_argument('--recommendations', 
                        help='Get model recommendations for a use case (e.g., "english_speech", "offline_processing")')
    parser.add_argument('--list-languages', action='store_true',
                        help='List all available languages and their codes')
    parser.add_argument('--list-models', action='store_true',
                        help='List all available models and their language support')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle utility commands
    if args.find_models:
        result = find_models_for_language(args.find_models)
        if 'error' in result:
            logging.error(result['error'])
            sys.exit(1)
        
        print(f"\n📋 Models supporting '{args.find_models}':")
        print(f"Language: {result['language']['display_name']}")
        print(f"Description: {result['language']['description']}")
        print(f"Language Codes: {', '.join(result['language']['language_codes'])}")
        print(f"Recommended Model: {result['recommendation']}")
        print(f"\nSupported Models:")
        for model in result['supported_models']:
            if model in result['model_details']:
                details = result['model_details'][model]
                print(f"  • {model}: {details['best_for']}")
        sys.exit(0)
    
    if args.find_languages:
        result = find_languages_for_model(args.find_languages)
        if 'error' in result:
            logging.error(result['error'])
            sys.exit(1)
        
        print(f"\n🌍 Languages supported by '{args.find_languages}':")
        print(f"Best for: {result['model']['best_for']}")
        print(f"Limitations: {result['model']['limitations']}")
        print(f"\nSupported Languages:")
        for lang in result['supported_languages']:
            print(f"  • {lang['language']} ({', '.join(lang['language_codes'])})")
            print(f"    {lang['description']}")
        sys.exit(0)
    
    if args.recommendations:
        result = get_quick_recommendations(args.recommendations)
        if 'error' in result:
            logging.error(result['error'])
            sys.exit(1)
        
        print(f"\n🎯 Recommendations for '{args.recommendations}':")
        print(f"Best Model: {result['best_model']}")
        print(f"Reason: {result['reason']}")
        print(f"All Recommended Models: {', '.join(result['recommended_models'])}")
        sys.exit(0)
    
    if args.list_languages:
        print("\n🌍 Available Languages:")
        for lang_key, lang_data in LANGUAGE_MAPPING.items():
            print(f"  • {lang_data['display_name']} ({lang_key})")
            print(f"    Codes: {', '.join(lang_data['language_codes'])}")
            print(f"    Models: {', '.join(lang_data['models'])}")
            print(f"    Description: {lang_data['description']}")
            print()
        sys.exit(0)
    
    if args.list_models:
        print("\n🤖 Available Models:")
        for model, details in MODEL_LANGUAGE_SUPPORT.items():
            print(f"  • {model}")
            print(f"    Best for: {details['best_for']}")
            print(f"    Languages: {', '.join(details['languages'])}")
            print(f"    Language Codes: {', '.join(details['language_codes'])}")
            print(f"    Limitations: {details['limitations']}")
            print()
        sys.exit(0)
    
    # Determine language
    dataset_config = DATASET_CONFIGS[args.dataset]
    language = args.language or dataset_config.get('language', 'english')
    
    # Validate model for language
    if not validate_model_for_language(args.model, language):
        supported_models = LANGUAGE_CONFIGS[language].get('models', [])
        logging.error(f"Model '{args.model}' does not support language '{language}'")
        logging.info(f"Supported models for {language}: {', '.join(supported_models)}")
        sys.exit(1)
    
    # Get model configuration
    model_config = get_model_config(args.model, language)

    # Add gcloud path for Google model
    if args.model in ['google', 'google_v2']:
        model_config['gcloud_path'] = args.gcloud_path
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log configuration
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Language: {language}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Output directory: {args.output_dir}")
    
    # Route to appropriate pipeline
    try:
        if args.dataset == 'librispeech':
            # Process LibriSpeech dataset
            process_librispeech(
                model_name=args.model,
                model_config=model_config,
                test_set=args.test_set,
                output_dir=args.output_dir
            )
            
        elif args.dataset == 'marathi-asr':
            # Process Marathi ASR dataset from HuggingFace
            process_marathi_asr(
                model_name=args.model,
                model_config=model_config,
                output_dir=args.output_dir
            )
            
        elif args.dataset == 'custom-csv':
            # Process custom CSV file
            if not args.csv:
                logging.error("CSV file path is required for custom-csv dataset")
                parser.print_help()
                sys.exit(1)
            
            if not os.path.exists(args.csv):
                logging.error(f"CSV file not found: {args.csv}")
                sys.exit(1)
            
            process_hinglish_csv(
                model_name=args.model,
                model_config=model_config,
                csv_path=args.csv,
                output_dir=args.output_dir,
                language=language
            )
            
        else:
            logging.error(f"Unknown dataset: {args.dataset}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=args.debug)
        sys.exit(1)


if __name__ == "__main__":
    main()