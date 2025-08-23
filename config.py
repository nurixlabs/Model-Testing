"""
Configuration for the STT evaluation pipeline
Loads sensitive information from environment variables
"""
import os

# Load environment variables from .env file


# Standard models available across all languages
STANDARD_MODELS = {
    'dolphin': {
        'display_name': 'Dolphin',
        'languages': ['english', 'hinglish', 'marathi'],
        'streaming': False
    },
    'whisper': {
        'display_name': 'Whisper Large v2',
        'languages': ['english', 'hinglish', 'marathi'],
        'streaming': False
    },
    'google': {
        'display_name': 'Google STT',
        'languages': ['english', 'hinglish', 'marathi'],
        'streaming': True
    },
    'google_v2': {
        'display_name': 'Google STT v2',
        'languages': ['english', 'hinglish', 'marathi'],
        'streaming': True
    },
    'aws': {
        'display_name': 'AWS STT',
        'languages': ['english', 'hinglish', 'marathi'],
        'streaming': True
    },
    'salad': {
        'display_name': 'Salad',
        'languages': ['english', 'hinglish', 'marathi'],
        'streaming': True
    },
    'deepgram_nova3': {
        'display_name': 'Deepgram Nova 3',
        'languages': ['english', 'hinglish', 'marathi'],
        'streaming': True
    },
    'deepgram_nova2': {
        'display_name': 'Deepgram Nova 2',
        'languages': ['english', 'hinglish', 'marathi'],
        'streaming': True
    },
    'sarvam': {
        'display_name': 'SARVAM',
        'languages': ['english', 'hinglish', 'marathi'],
        'streaming': True
    },
    'gladia': {
        'display_name': 'Gladia',
        'languages': ['english', 'hinglish', 'marathi'],
        'streaming': True
    },
    'azure': {
        'display_name': 'AZURE',
        'languages': ['english', 'hinglish', 'marathi'],
        'streaming': True
    },
    'assemblyai': {
        'display_name': 'AssemblyAI',
        'languages': ['english', 'hinglish', 'marathi'],
        'streaming': True
    }
}

# S3 Configuration
S3_CONFIG = {
    'bucket_name': os.getenv('S3_BUCKET_NAME', 'mlflow-artifacts-nurix'),
    'test_clean_prefix': os.getenv('S3_TEST_CLEAN_PREFIX', 'librispeech/test-clean/'),
    'test_other_prefix': os.getenv('S3_TEST_OTHER_PREFIX', 'librispeech/test-other/'),
}

# API Keys - Loaded from environment variables
API_KEYS = {
    'deepgram': os.getenv('DEEPGRAM_API_KEY'),
    'salad': os.getenv('SALAD_API_KEY'),
    'sarvam': os.getenv('SARVAM_API_KEY'),
    'google': os.getenv('GOOGLE_API_KEY'),
    'aws': None,  # AWS uses IAM roles or AWS CLI configuration
    'cartesia': os.getenv('CARTESIA_API_KEY'),
    'nvidia_parakeet': None,
    'assemblyai': os.getenv('ASSEMBLYAI_API_KEY'),
    'azure': os.getenv('AZURE_SPEECH_KEY'),
    'gladia': os.getenv('GLADIA_API_KEY'),
    'indic-conformer': None,
    'AI4bharat-conformer': None,
    'conformer_marathi': None,
}

# Model Configurations
MODEL_CONFIGS = {
    'dolphin': {
        'model_size': 'small',
        'model_dir': '~/.cache/dolphin',
        'device': 'cuda:0',
        'language': 'en',
        'region': 'IN',
    },
    'whisper': {
        'model_id': 'openai/whisper-large-v2',
        'device': None,  # Will be auto-detected (cuda, mps, or cpu)
        'batch_size': 1,
        'language': 'hi',
    },
    'google': {
        'language_code': 'en-IN',
        'project_id': os.getenv('GOOGLE_PROJECT_ID', 'train-453515'),
        'gcloud_path': './google-cloud-sdk/bin/gcloud',
    },
    'google_v2': {
        'project_id': os.getenv('GOOGLE_PROJECT_ID', 'train-453515'),
        'location': 'us-central1',
        'language_codes': ['en-IN'],
        'model': 'chirp_2',
        'enable_punctuation': True,
    },
    'aws': {
        'language_code': 'hi-IN',
        'max_concurrent_jobs': 90,
        'output_prefix': 'transcripts',
        'region': 'ap-south-1',
        's3_region': 'ap-south-1',
    },
    'salad': {
        'organization': 'nurix-ai',
        'api_base_url': 'https://api.salad.com',
    },
    'deepgram': {
        'model': 'nova-2',
        'language': 'hi',
        'punctuate': True,
        'smart_format': True,
        'location': 'us',
    },
    'deepgram_nova3': {
        'model': 'nova-3',
        'language': 'hi',
        'punctuate': True,
        'smart_format': True,
        'location': 'us',
    },
    'deepgram_nova2': {
        'model': 'nova-2',
        'language': 'hi',
        'punctuate': True,
        'smart_format': True,
        'location': 'us',
    },
    'sarvam': {
        'model': 'saarika:v2',
        'language_code': 'mr-IN',
        'max_retries': 5,
        'base_delay': 2.0,
        'max_delay': 60.0,
        'jitter': 0.25,
        'requests_per_minute': 10,
    },
    'gladia': {
        'model': 'solaria-1',
        'target_languages': ['en'],
    },
    'cartesia': {
        'model': 'ink-whisper',
        'language': 'en',
        'timestamp_granularities[]': 'word',
    },
    'nvidia_parakeet': {
        'model_id': 'nvidia/parakeet-tdt-0.6b-v2',
        'device': 'cuda' if os.getenv('USE_CUDA', 'true').lower() == 'true' else 'cpu',
    },
    'assemblyai': {
        'language': 'en',
        'punctuate': True,
        'format_text': True,
        'speaker_labels': False,
        'auto_highlights': False,
    },
    'azure': {
        'subscription_key': os.environ.get('AZURE_SPEECH_KEY') or os.getenv('AZURE_SPEECH_KEY'),
        'region': os.environ.get('AZURE_SPEECH_REGION') or os.getenv('AZURE_SPEECH_REGION'),
        'language': 'en-IN',  # Default to Indian English
        'enable_word_timing': True,
        'enable_punctuation': True,
    },
    'conformer_marathi': {
        'model_id': 'chintan-nurix/indicconformer_stt_multi_hybrid_rnnt_600m',
        'device': 'cuda' if os.getenv('USE_CUDA', 'true').lower() == 'true' else 'cpu',
        'language_code': 'mr',
        'decoding_method': 'ctc',
    },
}

# Output Configuration
OUTPUT_CONFIG = {
    'base_dir': os.getenv('OUTPUT_BASE_DIR', 'transcription_results'),
    'csv_filename': 'results.csv',
    'max_audio_duration': int(os.getenv('MAX_AUDIO_DURATION', '36000')),  # 10 hours in seconds
}

# Available Models
AVAILABLE_MODELS = [
    'dolphin', 'whisper', 'google', 'google_v2', 'aws', 'salad', 
    'deepgram', 'deepgram_nova3', 'deepgram_nova2', 'sarvam', 'gladia', 'cartesia', 'nvidia_parakeet', 
    'assemblyai', 'azure', 'conformer_marathi'
]

# Language configurations for different datasets
LANGUAGE_CONFIGS = {
    'english': {
        'models': ['dolphin', 'whisper', 'google', 'google_v2', 'aws', 'deepgram_nova3', 'deepgram_nova2', 'assemblyai', 'azure', 'salad', 'gladia', 'sarvam'],
        'default_language_code': 'en-US',
    },
    'hindi': {
        'models': ['whisper', 'google', 'google_v2', 'aws', 'deepgram_nova3', 'deepgram_nova2', 'sarvam', 'azure', 'gladia'],
        'default_language_code': 'hi-IN',
    },
    'marathi': {
        'models': ['whisper', 'google', 'google_v2', 'aws', 'deepgram_nova3', 'deepgram_nova2', 'sarvam', 'azure', 'gladia', 'indic-conformer', 'AI4bharat-conformer', 'conformer_marathi'],
        'default_language_code': 'mr',
    },
    'hinglish': {
        'models': ['whisper', 'google', 'google_v2', 'aws', 'deepgram_nova3', 'deepgram_nova2', 'azure', 'salad', 'gladia', 'sarvam'],
        'default_language_code': 'hi-IN',
    },
}

# Dataset configurations
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