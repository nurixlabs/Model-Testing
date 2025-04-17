# Configuration for the STT evaluation pipeline

# S3 Configuration
S3_CONFIG = {
    'bucket_name': 'mlflow-artifacts-nurix',
    'test_clean_prefix': 'librispeech/test-clean/',
    'test_other_prefix': 'librispeech/test-other/',
}

# API Keys - Replace with your actual keys or set via environment variables
API_KEYS = {
    'deepgram': None,
    'salad': None,
    'google': None,  # Set via environment variable GOOGLE_API_KEY
    'aws': None,     # Configure via AWS credentials
}

# Model Configurations
MODEL_CONFIGS = {
    'dolphin': {
        'model_size': 'small',
        'model_dir': '~/.cache/dolphin',
        'device': 'cuda',
        'language': 'en',
        'region': 'US',
    },
    'whisper': {
        'model_id': 'openai/whisper-large-v3-turbo',
        'device': None,  # Will be auto-detected (cuda, mps, or cpu)
        'batch_size': 1,
        'language': 'en',
    },
    'google': {
        'language_code': 'en-US',
    },
    'aws': {
        'language_code': 'en-US',
        'max_concurrent_jobs': 90,
        'output_prefix': 'transcripts',
    },
    'salad': {
        'organization': 'nurix-ai',
    },
    'deepgram': {
        'model': 'nova-3',
        'language': 'en',
        'punctuate': True,
        'smart_format': True,
    }
}

# Output Configuration
OUTPUT_CONFIG = {
    'base_dir': 'transcription_results',
    'csv_filename': 'results.csv',
    'max_audio_duration': 10 * 3600,  # 10 hours in seconds
}

# Available Models
AVAILABLE_MODELS = ['dolphin', 'whisper', 'google', 'aws', 'salad', 'deepgram']