# Speech-to-Text Evaluation Pipeline

A comprehensive evaluation framework for testing and comparing multiple Speech-to-Text (STT) models across different languages and datasets. This pipeline supports various commercial and open-source STT services, enabling systematic evaluation of transcription accuracy and performance.

## 🚀 Features

- **Multi-Model Support**: 14+ STT models including Whisper, Google, AWS, Azure, Deepgram, and more
- **Multi-Language Support**: English, Hindi, Marathi, and Hinglish (Hindi-English mixed)
- **Multiple Datasets**: LibriSpeech, Marathi ASR, and custom CSV datasets
- **Comprehensive Metrics**: WER (Word Error Rate), CER (Character Error Rate), and detailed analysis
- **Flexible Configuration**: Environment-based configuration for API keys and settings
- **Batch Processing**: Efficient processing of large audio datasets
- **Cloud Integration**: AWS S3 support for dataset storage and processing

## 📋 Available Models

| Model | Type | Languages | API Required |
|-------|------|-----------|--------------|
| **Whisper** | Open Source | All | ❌ |
| **Google Speech-to-Text** | Cloud | en-IN, hi-IN | ✅ |
| **Google Chirp 2** | Cloud | en-IN | ✅ |
| **AWS Transcribe** | Cloud | hi-IN | ✅ |
| **Azure Speech** | Cloud | en-US | ✅ |
| **Deepgram** | Cloud | hi, en | ✅ |
| **Deepgram Self-Hosted** | Self-Hosted | hi, en | ❌ |
| **Sarvam** | Cloud | mr-IN | ✅ |
| **AssemblyAI** | Cloud | en | ✅ |
| **Gladia** | Cloud | en | ✅ |
| **Cartesia** | Cloud | en | ✅ |
| **NVIDIA Parakeet** | Open Source | All | ❌ |
| **Dolphin** | Open Source | en | ❌ |
| **Conformer Marathi** | Open Source | mr | ❌ |

## 🌍 Supported Languages

### English
- **Models**: dolphin, whisper, google, aws, deepgram, assemblyai, azure
- **Default Code**: en-US
- **Datasets**: LibriSpeech

### Hindi
- **Models**: whisper, google, aws, deepgram, sarvam
- **Default Code**: hi-IN
- **Datasets**: Custom CSV

### Marathi
- **Models**: sarvam, conformer_marathi
- **Default Code**: mr
- **Datasets**: Marathi ASR (HuggingFace)

### Hinglish (Hindi-English Mixed)
- **Models**: whisper, google, deepgram
- **Default Code**: hi-IN
- **Datasets**: Custom CSV

## 📦 Installation

### Prerequisites

- Python 3.8+
- Git
- AWS CLI (for AWS services)
- Google Cloud SDK (for Google services)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Model-Testing
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# API Keys
DEEPGRAM_API_KEY=your_deepgram_key
SALAD_API_KEY=your_salad_key
SARVAM_API_KEY=your_sarvam_key
GOOGLE_API_KEY=your_google_key
CARTESIA_API_KEY=your_cartesia_key
ASSEMBLYAI_API_KEY=your_assemblyai_key
AZURE_SPEECH_KEY=your_azure_key
GLADIA_API_KEY=your_gladia_key

# Google Cloud
GOOGLE_PROJECT_ID=your_project_id

# AWS Configuration
S3_BUCKET_NAME=your_s3_bucket
S3_TEST_CLEAN_PREFIX=librispeech/test-clean/
S3_TEST_OTHER_PREFIX=librispeech/test-other/

# Output Configuration
OUTPUT_BASE_DIR=transcription_results
MAX_AUDIO_DURATION=36000

# CUDA Configuration
USE_CUDA=true
```

### AWS Configuration

For AWS services, configure your AWS credentials:

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region (e.g., us-east-1)
# Enter your output format (json)
```

### Google Cloud Configuration

For Google services, set up Google Cloud SDK:

```bash
# Download and install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate with Google Cloud
gcloud auth login
gcloud config set project your-project-id
```

## 🚀 Usage

### Basic Usage

```bash
python main.py --dataset <dataset> --model <model>
```

### Examples

#### 1. Process LibriSpeech with Whisper
```bash
python main.py --dataset librispeech --model whisper --test-set test-clean
```

#### 2. Process Marathi ASR with Sarvam
```bash
python main.py --dataset marathi-asr --model sarvam
```

#### 3. Process Custom CSV with Deepgram
```bash
python main.py --dataset custom-csv --model deepgram --csv path/to/file.csv --language hinglish
```

#### 4. Process with Google Speech-to-Text
```bash
python main.py --dataset librispeech --model google --test-set test-clean
```

#### 5. Process with AWS Transcribe
```bash
python main.py --dataset custom-csv --model aws --csv path/to/file.csv --language hindi
```

### Command Line Arguments

| Argument | Required | Description | Options |
|----------|----------|-------------|---------|
| `--dataset` | ✅ | Dataset to process | `librispeech`, `marathi-asr`, `custom-csv` |
| `--model` | ✅ | Model to use | See Available Models table |
| `--language` | ❌ | Override language | `english`, `hindi`, `marathi`, `hinglish` |
| `--test-set` | ❌ | LibriSpeech test set | `test-clean`, `test-other`, `both` |
| `--csv` | ❌ | Path to CSV file (for custom-csv) | File path |
| `--output-dir` | ❌ | Output directory | Directory path |
| `--gcloud-path` | ❌ | Path to gcloud executable | File path |
| `--debug` | ❌ | Enable debug logging | Flag |

## 📊 Output

The pipeline generates comprehensive evaluation results:

### Directory Structure
```
transcription_results/
├── {model_name}/
│   ├── {dataset_name}/
│   │   ├── results.csv          # Detailed results
│   │   ├── metrics.json         # Summary metrics
│   │   └── logs/                # Processing logs
```

### Metrics Included
- **WER (Word Error Rate)**: Word-level accuracy
- **CER (Character Error Rate)**: Character-level accuracy
- **Processing Time**: Time taken for transcription
- **Audio Duration**: Length of processed audio
- **Error Analysis**: Detailed error breakdown

## 🔧 Advanced Configuration

### Model-Specific Settings

Each model can be configured in `config.py`:

```python
MODEL_CONFIGS = {
    'whisper': {
        'model_id': 'openai/whisper-large-v2',
        'device': None,  # Auto-detect
        'batch_size': 1,
        'language': 'hi',
    },
    'aws': {
        'language_code': 'hi-IN',
        'max_concurrent_jobs': 90,
        'region': 'us-east-1',
    },
    # ... more models
}
```

### Custom Datasets

To add custom datasets:

1. **Create a new pipeline** in `pipelines/`
2. **Add dataset configuration** in `config.py`
3. **Update main.py** to handle the new dataset

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all required API keys are set in `.env`
   - Check API key validity and permissions

2. **AWS Configuration Issues**
   - Verify AWS CLI is installed and configured
   - Check IAM permissions for Transcribe and S3

3. **Google Cloud Issues**
   - Ensure Google Cloud SDK is installed
   - Verify project ID and authentication

4. **CUDA/GPU Issues**
   - Set `USE_CUDA=false` in `.env` for CPU-only mode
   - Check CUDA installation and compatibility

### Debug Mode

Enable debug logging for detailed error information:

```bash
python main.py --dataset librispeech --model whisper --debug
```

## 📈 Performance Tips

1. **Batch Processing**: Use appropriate batch sizes for your hardware
2. **Concurrent Jobs**: Adjust `max_concurrent_jobs` for cloud services
3. **Audio Duration Limits**: Set `MAX_AUDIO_DURATION` to control processing time
4. **GPU Usage**: Enable CUDA for faster processing with supported models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LibriSpeech](https://www.openslr.org/12/) for English audio dataset
- [Marathi ASR Dataset](https://huggingface.co/datasets/TheAIchemist13/marathi_asr_dataset) for Marathi audio dataset
- All the STT service providers for their APIs and models

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration examples

---

**Note**: This pipeline is designed for research and evaluation purposes. Please ensure compliance with the terms of service for all third-party APIs and services used. 