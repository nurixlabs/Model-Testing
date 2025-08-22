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
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename


# Try to import model-related modules
try:
    from models.model_factory import get_model
    from config import MODEL_CONFIGS, AVAILABLE_MODELS, LANGUAGE_CONFIGS, STANDARD_MODELS
    MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import model modules: {e}")
    MODELS_AVAILABLE = False
    # Define some defaults if imports fail
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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store running tasks
running_tasks = {}

# Initial STT data - this is your actual data
try:
    with open('/home/azureuser/test-pipelines/Model-Testing/dashboard/src/sttData.json', 'r') as f:
        STT_DATA = json.load(f)
except FileNotFoundError:
    logger.warning("sttData.json not found, using empty data")
    STT_DATA = {}

# Store the current data in memory
current_stt_data = STT_DATA.copy()


def merge_available_models_with_results(stt_data):
    """
    Merge the available standard models with existing test results.
    This ensures all standard models appear in the dashboard even if not tested yet.
    """
    merged_data = stt_data.copy()
    
    # Ensure all languages exist
    for language in ['english', 'hinglish', 'marathi']:
        if language not in merged_data:
            merged_data[language] = {
                'dataset': get_dataset_description(language),
                'models': []
            }
    
    # For each language, add standard models if they don't exist
    for language in ['english', 'hinglish', 'marathi']:
        existing_model_names = {model['name'] for model in merged_data[language].get('models', [])}
        
        # Add standard models that are available for this language
        for model_key, model_info in STANDARD_MODELS.items():
            if language in model_info.get('languages', []):
                display_name = model_info['display_name']
                
                # Skip if already exists in test results
                if display_name in existing_model_names:
                    continue
                
                # Add model with placeholder data
                model_entry = {
                    'name': display_name,
                    'streaming': model_info.get('streaming', False),
                    'status': 'available',  # Indicates model is available but not tested
                    'cost_batch': 'Not tested',
                    'cost_streaming': 'Not tested',
                    'latency_batch': None,
                    'latency_streaming': None
                }
                
                # Add language-specific placeholder metrics
                if language == 'english':
                    model_entry.update({
                        'wer_clean': None,
                        'cer_clean': None,
                        'wer_other': None,
                        'cer_other': None
                    })
                elif language == 'hinglish':
                    model_entry.update({
                        'score': None
                    })
                elif language == 'marathi':
                    model_entry.update({
                        'wer': None,
                        'cer': None
                    })
                
                merged_data[language]['models'].append(model_entry)
    
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
        'Google STT v2': 'google_v2',
        'AWS STT': 'aws',
        'Salad': 'salad',
        'Deepgram Nova 3': 'deepgram_nova3',
        'Deepgram Nova 2': 'deepgram_nova2',
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


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get current STT results with all available models."""
    logger.info("Sending STT data to frontend")
    # Merge available models with existing results
    merged_data = merge_available_models_with_results(current_stt_data)
    return jsonify(merged_data)


@app.route('/api/available-models', methods=['GET'])
def get_available_models():
    """Get list of available models for a language."""
    language = request.args.get('language', 'english')
    
    available = []
    for model_key, model_info in STANDARD_MODELS.items():
        if language in model_info.get('languages', []):
            available.append({
                'key': model_key,
                'display_name': model_info['display_name'],
                'streaming': model_info.get('streaming', False)
            })
    
    return jsonify({'models': available})


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
                    
                    # Check if model exists in configs
                    if model_key not in MODEL_CONFIGS:
                        logger.warning(f"Model {model_key} not found in MODEL_CONFIGS")
                        # Try to use base deepgram config for Nova variants
                        if model_key.startswith('deepgram'):
                            base_config = MODEL_CONFIGS.get('deepgram', {}).copy()
                            if 'nova3' in model_key:
                                base_config['model'] = 'nova-3'
                            elif 'nova2' in model_key:
                                base_config['model'] = 'nova-2'
                            model_config = base_config
                        else:
                            raise ValueError(f"No configuration found for {model_key}")
                    else:
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
                    
                    # Handle special model cases
                    if model_key.startswith('deepgram'):
                        # For deepgram variants, always use the base deepgram model class
                        actual_model_key = 'deepgram'
                    else:
                        actual_model_key = model_key
                    
                    # Initialize and load model
                    logger.info(f"Initializing {actual_model_key} with config: {model_config}")
                    model = get_model(actual_model_key, model_config)
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
                    # Use mock transcription
                    if language == 'english':
                        transcription = f"Mock transcription from {model_display_name}: Hello, this is a test audio file."
                    elif language == 'hinglish':
                        transcription = f"Mock from {model_display_name}: Namaste, yeh ek test audio file hai."
                    else:
                        transcription = f"Mock from {model_display_name}: नमस्कार, ही एक चाचणी ऑडिओ फाइल आहे."
                
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
        
        # Clean up old files
        try:
            import glob
            old_files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
            current_time = time.time()
            for old_file in old_files:
                if os.path.getmtime(old_file) < current_time - 3600:
                    os.remove(old_file)
        except:
            pass
        
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error in audio test: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test-csv', methods=['POST'])
def test_csv():
    """Test a model on CSV dataset"""
    try:
        if 'csv' not in request.files:
            return jsonify({'error': 'No CSV file provided'}), 400
        
        csv_file = request.files['csv']
        model_name = request.form.get('model')
        language = request.form.get('language', 'english')
        dataset_name = request.form.get('dataset_name', f'custom_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        if not model_name:
            return jsonify({'error': 'No model selected'}), 400
        
        # Save CSV file
        csv_filename = secure_filename(csv_file.filename)
        csv_path = os.path.join(UPLOAD_FOLDER, f"{time.time()}_{csv_filename}")
        csv_file.save(csv_path)
        
        # Create task ID
        task_id = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize task
        running_tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'dataset_name': dataset_name,
            'started': datetime.now().isoformat()
        }
        
        # Run test in background thread
        def run_csv_test():
            try:
                # TODO: Implement actual CSV processing with the selected model
                # For now, simulate processing
                for i in range(10):
                    time.sleep(0.5)
                    running_tasks[task_id]['progress'] = (i + 1) * 10
                
                # Generate mock metrics based on model
                model_key = get_model_key_from_display_name(model_name)
                
                if language == 'english':
                    metrics = {
                        'name': model_name,
                        'wer_clean': 0.15 + (hash(model_name) % 10) * 0.01,
                        'cer_clean': 0.04 + (hash(model_name) % 10) * 0.005,
                        'latency_batch': 2.0 + (hash(model_name) % 5),
                        'streaming': STANDARD_MODELS.get(model_key, {}).get('streaming', True)
                    }
                elif language == 'hinglish':
                    metrics = {
                        'name': model_name,
                        'score': 3.5 + (hash(model_name) % 10) * 0.15,
                        'latency_batch': 2.0 + (hash(model_name) % 5),
                        'streaming': STANDARD_MODELS.get(model_key, {}).get('streaming', True)
                    }
                else:
                    metrics = {
                        'name': model_name,
                        'wer': 0.25 + (hash(model_name) % 10) * 0.02,
                        'cer': 0.08 + (hash(model_name) % 10) * 0.01,
                        'latency_batch': 2.0 + (hash(model_name) % 5),
                        'streaming': STANDARD_MODELS.get(model_key, {}).get('streaming', True)
                    }
                
                # Update the model in current data if it exists
                global current_stt_data
                if language in current_stt_data:
                    # Find and update existing model or add new
                    model_found = False
                    for idx, model in enumerate(current_stt_data[language]['models']):
                        if model['name'] == model_name:
                            current_stt_data[language]['models'][idx].update(metrics)
                            model_found = True
                            break
                    
                    if not model_found:
                        current_stt_data[language]['models'].append(metrics)
                
                # Mark task as completed
                running_tasks[task_id]['status'] = 'completed'
                running_tasks[task_id]['progress'] = 100
                
            except Exception as e:
                logger.error(f"CSV test failed: {e}")
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
            'message': f'CSV testing started for {model_name}'
        })
        
    except Exception as e:
        logger.error(f"Error starting CSV test: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get status of a running task"""
    if task_id not in running_tasks:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(running_tasks[task_id])


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_available': MODELS_AVAILABLE,
        'model_count': len(AVAILABLE_MODELS) if MODELS_AVAILABLE else 0,
        'standard_models': len(STANDARD_MODELS)
    })


if __name__ == '__main__':
    logger.info("Starting STT Dashboard API server...")
    logger.info(f"Data loaded: {len(current_stt_data)} languages")
    logger.info(f"Standard models configured: {len(STANDARD_MODELS)}")
    app.run(debug=True, port=5000, host='0.0.0.0')