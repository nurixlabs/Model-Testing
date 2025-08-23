#!/usr/bin/env python3
"""
Flask API server for STT Model Testing Dashboard
"""
import os
import json
import logging
import tempfile
import threading
import time
import hashlib
import glob
import csv as csv_module
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from dotenv import load_dotenv

load_dotenv()

os.environ['AWS_PROFILE'] = 'Power-root'

# Initialize AWS Secrets Manager early
try:
    from secrets_manager import initialize_secrets, get_secret
    logger = logging.getLogger(__name__)
    logger.info("Initializing AWS Secrets Manager...")
    if initialize_secrets():
        logger.info("✅ AWS Secrets Manager initialized successfully")
    else:
        logger.warning("⚠️ AWS Secrets Manager initialization failed, falling back to environment variables")
except ImportError as e:
    logger.warning(f"Secrets manager not available: {e}")
    get_secret = os.environ.get

# Import pipeline functions
from pipelines.english_pipeline_updated import process_english
from pipelines.marathi_pipeline_updated import process_marathi
from pipelines.hinglish_pipeline_updated import process_hinglish

# Try to import model-related modules
try:
    from config import STANDARD_MODELS, MODEL_CONFIGS, AVAILABLE_MODELS, LANGUAGE_CONFIGS
    from models.model_factory import get_model
    MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import config modules: {e}")
    MODELS_AVAILABLE = False
    MODEL_CONFIGS = {}
    AVAILABLE_MODELS = []
    LANGUAGE_CONFIGS = {}
    STANDARD_MODELS = {}

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths
UPLOAD_FOLDER = 'temp_uploads'
RESULTS_FOLDER = 'transcription_results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Store running tasks
running_tasks = {}

# STT data file path
STT_DATA_FILE = 'dashboard/src/sttData.json'

# Load initial STT data
def load_stt_data():
    """Load STT data from file"""
    try:
        with open(STT_DATA_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"STT data file not found at {STT_DATA_FILE}, using default data")
        return {
            "english": {"dataset": "Librispeech Dataset (US English Benchmark)", "models": []},
            "hinglish": {"dataset": "In house Hinglish dataset (cult and youtube)", "models": []},
            "marathi": {"dataset": "TheAIchemist13/marathi_asr_dataset", "models": []}
        }

def save_stt_data(data):
    """Save STT data to file"""
    try:
        with open(STT_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"✅ STT data saved to {STT_DATA_FILE}")
    except Exception as e:
        logger.error(f"Failed to save STT data: {e}")

# Load initial data
current_stt_data = load_stt_data()

# Track processed CSV files (hash -> dataset info)
csv_datasets = {}
CSV_TRACKING_FILE = 'csv_datasets.json'

def load_csv_tracking():
    """Load CSV tracking data"""
    global csv_datasets
    try:
        with open(CSV_TRACKING_FILE, 'r') as f:
            csv_datasets = json.load(f)
    except FileNotFoundError:
        csv_datasets = {}

def save_csv_tracking():
    """Save CSV tracking data"""
    try:
        with open(CSV_TRACKING_FILE, 'w') as f:
            json.dump(csv_datasets, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save CSV tracking: {e}")

load_csv_tracking()

def get_csv_hash(file_path):
    """Get hash of CSV file for tracking"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def merge_available_models_with_results(stt_data):
    """Merge available standard models with existing test results."""
    merged_data = stt_data.copy()
    
    # Add custom CSV datasets with language-specific keys
    for lang_csv_hash, csv_info in csv_datasets.items():
        lang = csv_info['language']
        dataset_key = csv_info['dataset_key']
        
        # Check if this dataset already exists in the language
        if dataset_key not in merged_data:
            merged_data[dataset_key] = {
                'dataset': csv_info['dataset_name'],
                'models': csv_info.get('models', []),
                'is_custom': True,
                'csv_hash': lang_csv_hash,
                'language': lang
            }
    
    return merged_data

def get_dataset_description(language):
    """Get the dataset description for a language."""
    descriptions = {
        'english': 'Librispeech Dataset (US English Benchmark)',
        'hinglish': 'In house Hinglish dataset (cult and youtube)',
        'marathi': 'TheAIchemist13/marathi_asr_dataset'
    }
    return descriptions.get(language, f'{language.title()} Dataset')

def get_model_key_from_display_name(display_name):
    """Map display name back to model key for processing."""
    name_to_key = {
        'Dolphin': 'dolphin',
        'Whisper Large v2': 'whisper',
        'Whisper': 'whisper',
        'Google STT': 'google',
        'Google STT v1': 'google',
        'Google STT v2': 'google_v2',
        'Gemini 2.5 Pro': 'gemini',
        'AWS STT': 'aws',
        'AZURE STT': 'azure',
        'Salad': 'salad',
        'Deepgram Nova 3': 'deepgram_nova3',
        'Deepgram Nova 2': 'deepgram_nova2',
        'Sarvam': 'sarvam',
        'SARVAM': 'sarvam',
        'Gladia': 'gladia',
        'AZURE': 'azure',
        'AssemblyAI': 'assemblyai',
        'Cartesia Ink Whisper': 'cartesia',
        'IndicConformer Multi (RNNT)': 'conformer_marathi',
        'IndicConformer Multi (CTC)': 'conformer_marathi',
        'IndicConformer Large (RNNT)': 'conformer_marathi',
        'IndicConformer Large (CTC)': 'conformer_marathi'
    }
    return name_to_key.get(display_name, display_name.lower().replace(' ', '_'))

def process_pipeline_results(language, model_name, output_dir, csv_hash, dataset_name, model_key, is_default_dataset=False):
    """Process pipeline results and update STT data"""
    global current_stt_data, csv_datasets
    
    # Look for metrics.json in the expected pipeline output structure
    metrics_file = None
    
    # Try multiple possible paths using glob patterns
    possible_patterns = [
        os.path.join(output_dir, model_key, "*", "metrics.json"),
        os.path.join(output_dir, "*", "metrics.json"),
        os.path.join(output_dir, "metrics.json"),
    ]
    
    for pattern in possible_patterns:
        matches = glob.glob(pattern)
        if matches:
            metrics_file = matches[0]
            logger.info(f"Found metrics file using pattern {pattern}: {metrics_file}")
            break
    
    # If still not found, walk the directory
    if not metrics_file:
        for root, dirs, files in os.walk(output_dir):
            if 'metrics.json' in files:
                metrics_file = os.path.join(root, 'metrics.json')
                logger.info(f"Found metrics file by walking directory: {metrics_file}")
                break
    
    # If no metrics.json, try to calculate from results.csv
    if not metrics_file:
        logger.warning(f"No metrics.json found, attempting to calculate from results.csv")
        results_csv = None
        
        # Find results.csv
        for root, dirs, files in os.walk(output_dir):
            if 'results.csv' in files:
                results_csv = os.path.join(root, 'results.csv')
                logger.info(f"Found results.csv at: {results_csv}")
                break
        
        if results_csv:
            # Calculate metrics from CSV
            wer_scores = []
            cer_scores = []
            llm_scores = []
            
            with open(results_csv, 'r', encoding='utf-8') as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    if 'wer' in row and row['wer']:
                        try:
                            wer_scores.append(float(row['wer']))
                        except:
                            pass
                    if 'cer' in row and row['cer']:
                        try:
                            cer_scores.append(float(row['cer']))
                        except:
                            pass
                    if 'llm_score' in row and row['llm_score']:
                        try:
                            llm_scores.append(float(row['llm_score']))
                        except:
                            pass
            
            # Create metrics based on what we found
            metrics = {}
            if wer_scores:
                metrics['avg_wer'] = sum(wer_scores) / len(wer_scores)
            if cer_scores:
                metrics['avg_cer'] = sum(cer_scores) / len(cer_scores)
            if llm_scores:
                metrics['average_score'] = sum(llm_scores) / len(llm_scores)
                metrics['avg_llm_score'] = metrics['average_score']
            
            metrics['num_files'] = max(len(wer_scores), len(cer_scores), len(llm_scores), 1)
            
            logger.info(f"Calculated metrics from results.csv: {metrics}")
        else:
            logger.error(f"No metrics.json or results.csv found in {output_dir}")
            # List directory structure for debugging
            for root, dirs, files in os.walk(output_dir):
                logger.info(f"  {root}: dirs={dirs}, files={files[:5]}")  # Limit files shown
            return None
    else:
        logger.info(f"Loading metrics from: {metrics_file}")
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    
    logger.info(f"Processing metrics: {metrics}")
    
    # Create model entry based on language
    model_entry = {
        'name': model_name,
        'streaming': STANDARD_MODELS.get(model_key, {}).get('streaming', False),
        'cost_batch': 'Tested',
        'cost_streaming': 'Tested'
    }
    
    if language == 'english':
        model_entry.update({
            'wer_clean': metrics.get('avg_wer', 0),
            'cer_clean': metrics.get('avg_cer', 0),
            'wer_other': metrics.get('avg_wer', 0) * 1.2,  # Estimate
            'cer_other': metrics.get('avg_cer', 0) * 1.2  # Estimate
        })
    elif language == 'hinglish':
        # For Hinglish, the metric might be 'average_score' or 'avg_llm_score'
        score = metrics.get('average_score') or metrics.get('avg_llm_score', 3.0)
        model_entry.update({
            'score': score
        })
    elif language == 'marathi':
        model_entry.update({
            'wer': metrics.get('avg_wer', 0),
            'cer': metrics.get('avg_cer', 0)
        })
    
    # Handle default dataset differently
    if is_default_dataset and language == 'hinglish':
        # For default dataset, update the main language entry
        dataset_key = 'hinglish'
        lang_csv_hash = None  # Don't track in csv_datasets
    else:
        # Create language-specific dataset key for custom CSV
        dataset_key = f"{language}_csv_{csv_hash[:8]}"
        lang_csv_hash = f"{language}_{csv_hash}"
    
    # Update CSV tracking with language-specific key (only for custom datasets)
    if lang_csv_hash and lang_csv_hash in csv_datasets:
        # Update existing entry
        existing_models = csv_datasets[lang_csv_hash].get('models', [])
        model_found = False
        for idx, existing_model in enumerate(existing_models):
            if existing_model['name'] == model_name:
                existing_models[idx] = model_entry
                model_found = True
                break
        if not model_found:
            existing_models.append(model_entry)
        csv_datasets[lang_csv_hash]['models'] = existing_models
    elif lang_csv_hash:
        # Create new entry (only for custom datasets)
        csv_datasets[lang_csv_hash] = {
            'dataset_key': dataset_key,
            'dataset_name': dataset_name,
            'language': language,
            'models': [model_entry],
            'upload_time': datetime.now().isoformat()
        }
    
    # Update current STT data
    if dataset_key not in current_stt_data:
        current_stt_data[dataset_key] = {
            'dataset': dataset_name,
            'models': [],
            'is_custom': True,
            'language': language
        }
    
    # Check if model already exists and update or add
    model_found = False
    for idx, model in enumerate(current_stt_data[dataset_key]['models']):
        if model['name'] == model_name:
            current_stt_data[dataset_key]['models'][idx] = model_entry
            model_found = True
            break
    
    if not model_found:
        current_stt_data[dataset_key]['models'].append(model_entry)
    
    # Do NOT update in the standard language section - keep datasets separate
    
    # Save updated data
    save_stt_data(current_stt_data)
    save_csv_tracking()
    
    return metrics

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get current STT results with all available models."""
    logger.info("Sending STT data to frontend")
    merged_data = merge_available_models_with_results(current_stt_data)
    return jsonify(merged_data)

@app.route('/api/available-models', methods=['GET'])
def get_available_models():
    """Get list of available models for testing."""
    # Fixed list of models available for all languages
    fixed_models = [
        {'key': 'deepgram_nova3', 'display_name': 'Deepgram Nova 3', 'streaming': True},
        {'key': 'deepgram_nova2', 'display_name': 'Deepgram Nova 2', 'streaming': True},
        {'key': 'whisper', 'display_name': 'Whisper', 'streaming': False},
        {'key': 'google_v2', 'display_name': 'Google STT v2', 'streaming': True},
        {'key': 'aws', 'display_name': 'AWS STT', 'streaming': True},
        {'key': 'azure', 'display_name': 'AZURE STT', 'streaming': True},
        {'key': 'gladia', 'display_name': 'Gladia', 'streaming': True},
        {'key': 'assemblyai', 'display_name': 'AssemblyAI', 'streaming': True},
        {'key': 'sarvam', 'display_name': 'Sarvam', 'streaming': True}
    ]
    
    return jsonify({'models': fixed_models})

@app.route('/api/test-csv', methods=['POST'])
def test_csv():
    """Test a model on CSV dataset using actual pipelines"""
    try:
        # Check if using default dataset for Hinglish
        use_default = request.form.get('use_default') == 'true'
        language = request.form.get('language', 'english')
        model_name = request.form.get('model')
        
        if not model_name:
            return jsonify({'error': 'No model selected'}), 400
        
        if use_default and language == 'hinglish':
            # Use the default Hinglish dataset
            csv_path = os.path.join('datasets', 'hinglish.csv')
            dataset_name = 'hinglish'  # This will update the main hinglish dataset in sttData.json
            
            if not os.path.exists(csv_path):
                return jsonify({'error': 'Default Hinglish dataset not found at datasets/hinglish.csv'}), 404
        else:
            # Handle uploaded CSV
            if 'csv' not in request.files:
                return jsonify({'error': 'No CSV file provided'}), 400
            
            csv_file = request.files['csv']
            dataset_name = request.form.get('dataset_name', f'custom_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            
            # Save CSV file
            csv_filename = secure_filename(csv_file.filename)
            csv_path = os.path.join(UPLOAD_FOLDER, f"{time.time()}_{csv_filename}")
            csv_file.save(csv_path)
        
        # Get CSV hash to track duplicates
        csv_hash = get_csv_hash(csv_path)
        
        # Create language-specific hash
        lang_csv_hash = f"{language}_{csv_hash}"
        
        # Check if this CSV was processed before for this language
        existing_dataset = csv_datasets.get(lang_csv_hash)
        if existing_dataset:
            dataset_name = existing_dataset['dataset_name']
            logger.info(f"CSV already processed before for {language} as: {dataset_name}")
        
        # Create task ID
        task_id = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize task
        running_tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'dataset_name': dataset_name,
            'started': datetime.now().isoformat(),
            'csv_hash': csv_hash,
            'language': language
        }
        
        # Get model key from display name
        model_key = get_model_key_from_display_name(model_name)
        
        # Get model config
        model_config = MODEL_CONFIGS.get(model_key, {}).copy()
        
        # Apply language-specific configuration
        if language == 'english':
            if 'language_code' in model_config:
                model_config['language_code'] = 'en-US'
            if 'language' in model_config:
                model_config['language'] = 'en'
        elif language == 'hinglish':
            if 'language_code' in model_config:
                model_config['language_code'] = 'hi-IN'
            if 'language' in model_config:
                model_config['language'] = 'hi'
        elif language == 'marathi':
            if 'language_code' in model_config:
                model_config['language_code'] = 'mr-IN'
            if 'language' in model_config:
                model_config['language'] = 'mr'
        
        # Run test in background thread
        def run_csv_test():
            try:
                running_tasks[task_id]['progress'] = 10
                
                # Set output directory - just use RESULTS_FOLDER as base
                output_dir = RESULTS_FOLDER
                
                # Call appropriate pipeline based on language
                logger.info(f"Starting {language} pipeline for {model_name} on {dataset_name}")
                logger.info(f"Output directory: {output_dir}")
                logger.info(f"CSV path: {csv_path}")
                
                if language == 'english':
                    process_english(
                        model_name=model_key,
                        model_config=model_config,
                        output_dir=output_dir,
                        dataset_key='custom-csv',
                        csv_path=csv_path
                    )
                elif language == 'marathi':
                    process_marathi(
                        model_name=model_key,
                        model_config=model_config,
                        output_dir=output_dir,
                        dataset_key='custom-csv',
                        csv_path=csv_path
                    )
                elif language in ('hinglish', 'hindi'):
                    process_hinglish(
                        model_name=model_key,
                        model_config=model_config,
                        output_dir=output_dir,
                        dataset_key='custom-csv',
                        csv_path=csv_path,
                        language=language
                    )
                else:
                    raise ValueError(f"Unsupported language: {language}")
                
                running_tasks[task_id]['progress'] = 80
                
                # Process results and update STT data
                is_default = use_default and language == 'hinglish'
                metrics = process_pipeline_results(
                    language, 
                    model_name, 
                    output_dir, 
                    csv_hash,
                    dataset_name,
                    model_key,
                    is_default_dataset=is_default
                )
                
                running_tasks[task_id]['progress'] = 100
                running_tasks[task_id]['status'] = 'completed'
                running_tasks[task_id]['metrics'] = metrics
                
                logger.info(f"CSV test completed for {dataset_name}")
                
            except Exception as e:
                logger.error(f"CSV test failed: {e}", exc_info=True)
                running_tasks[task_id]['status'] = 'failed'
                running_tasks[task_id]['error'] = str(e)
            finally:
                try:
                    os.remove(csv_path)
                except:
                    pass
        
        # Start background thread
        thread = threading.Thread(target=run_csv_test)
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'status': 'started',
            'message': f'CSV testing started for {model_name} on {dataset_name}'
        })
        
    except Exception as e:
        logger.error(f"Error starting CSV test: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get status of a running task"""
    if task_id not in running_tasks:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(running_tasks[task_id])

@app.route('/api/audio-test', methods=['POST'])
def test_audio():
    """Test multiple models on uploaded audio file"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        models = request.form.getlist('models[]')
        language = request.form.get('language', 'english')
        
        if not models:
            return jsonify({'error': 'No models selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(UPLOAD_FOLDER, f"{time.time()}_{filename}")
        audio_file.save(audio_path)
        
        results = []
        
        for model_display_name in models:
            try:
                start_time = time.time()
                
                if MODELS_AVAILABLE:
                    # Get actual model key from display name
                    model_key = get_model_key_from_display_name(model_display_name)
                    
                    # Get model config
                    model_config = MODEL_CONFIGS.get(model_key, {}).copy()
                    
                    # Set language-specific parameters
                    if language == 'english':
                        if 'language_code' in model_config:
                            model_config['language_code'] = 'en-US'
                        if 'language' in model_config:
                            model_config['language'] = 'en'
                    elif language == 'hinglish':
                        if 'language_code' in model_config:
                            model_config['language_code'] = 'hi-IN'
                        if 'language' in model_config:
                            model_config['language'] = 'hi'
                    elif language == 'marathi':
                        if 'language_code' in model_config:
                            model_config['language_code'] = 'mr-IN'
                        if 'language' in model_config:
                            model_config['language'] = 'mr'
                    
                    # Initialize and load model
                    logger.info(f"Initializing {model_key} with config: {model_config}")
                    model = get_model(model_key, model_config)
                    model.load()
                    
                    # Transcribe the audio
                    logger.info(f"Transcribing with {model_display_name}...")
                    result = model.transcribe(audio_path)
                    
                    # Extract transcription text
                    transcription = result.get('text', '') or result.get('transcript', '') or result.get('transcription', '')
                    
                    if not transcription:
                        logger.warning(f"No transcription returned from {model_display_name}")
                        transcription = "No transcription returned"
                else:
                    # Model not available
                    raise Exception("Model not available. Please ensure models are properly configured.")
                
                processing_time = time.time() - start_time
                
                results.append({
                    'model': model_display_name,
                    'transcription': transcription,
                    'processingTime': processing_time
                })
                
                logger.info(f"Successfully transcribed with {model_display_name}")
                
            except Exception as e:
                logger.error(f"Error testing {model_display_name}: {e}")
                results.append({
                    'model': model_display_name,
                    'error': str(e),
                    'transcription': f"Error: {str(e)}",
                    'processingTime': 0
                })
        
        # Clean up
        try:
            os.remove(audio_path)
        except:
            pass
        
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error in audio test: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def simple_health_check():
    """Simple health check endpoint for ALB"""
    return jsonify({'status': 'healthy'})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Detailed health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_available': MODELS_AVAILABLE,
        'model_count': len(AVAILABLE_MODELS) if MODELS_AVAILABLE else 0,
        'standard_models': len(STANDARD_MODELS),
        'custom_datasets': len(csv_datasets)
    })

# Static files for React frontend
@app.route('/static/css/<path:filename>')
def serve_static_css(filename):
    """Serve CSS files for React frontend"""
    try:
        return send_from_directory('dashboard/build/static/css', filename)
    except FileNotFoundError:
        logger.warning(f"CSS file not found: {filename}")
        return "File not found", 404

@app.route('/static/js/<path:filename>')
def serve_static_js(filename):
    """Serve JS files for React frontend"""
    try:
        return send_from_directory('dashboard/build/static/js', filename)
    except FileNotFoundError:
        logger.warning(f"JS file not found: {filename}")
        return "File not found", 404

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve React frontend for all non-API routes"""
    if path.startswith('api/'):
        return "API endpoint not found", 404
    
    try:
        if path and not path.startswith('api'):
            try:
                return send_from_directory('dashboard/build', path)
            except FileNotFoundError:
                pass
        
        return send_from_directory('dashboard/build', 'index.html')
    except FileNotFoundError:
        logger.error("React build files not found.")
        return """
        <h1>STT Dashboard</h1>
        <p>Frontend build files not found. The React app needs to be built first.</p>
        <p>Available API endpoints:</p>
        <ul>
            <li><a href="/api/results">/api/results</a> - Get test results</li>
            <li><a href="/api/health">/api/health</a> - Health check</li>
        </ul>
        """, 200

if __name__ == '__main__':
    logger.info("Starting STT Dashboard API server...")
    logger.info(f"Data loaded: {len(current_stt_data)} datasets")
    logger.info(f"Standard models configured: {len(STANDARD_MODELS)}")
    logger.info(f"Custom CSV datasets tracked: {len(csv_datasets)}")
    app.run(debug=True, port=5000, host='0.0.0.0')