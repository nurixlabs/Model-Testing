import React, { useState, useEffect, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter } from 'recharts';
import './STTDashboard.css';

const STTDashboard = () => {
  const [selectedLanguage, setSelectedLanguage] = useState('english');
  const [selectedModel1, setSelectedModel1] = useState('');
  const [selectedModel2, setSelectedModel2] = useState('');
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedDataset, setSelectedDataset] = useState('default');
  
  // Testing functionality states
  const [selectedTestModel, setSelectedTestModel] = useState('');
  const [selectedTestModels, setSelectedTestModels] = useState([]);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedAudioFile, setUploadedAudioFile] = useState(null);
  const [showTestingModal, setShowTestingModal] = useState(false);
  const [testingProgress, setTestingProgress] = useState(0);
  const [audioTestResults, setAudioTestResults] = useState(null);
  const [customDatasets, setCustomDatasets] = useState([]);
  const [sttData, setSttData] = useState({
    english: { dataset: 'Loading...', models: [] },
    marathi: { dataset: 'Loading...', models: [] },
    hinglish: { dataset: 'Loading...', models: [] }
  });
  const [refreshKey, setRefreshKey] = useState(0);
  const [availableTestModels, setAvailableTestModels] = useState([]);

  // Load STT data from backend
  const loadData = React.useCallback(() => {
    console.log('Fetching data from backend...');
    fetch('/api/results')
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then(data => {
        console.log('Data received:', data);
        setSttData(data);
        
        // Extract custom datasets for the current language
        const customKeys = Object.keys(data).filter(key => {
          const dataset = data[key];
          return (dataset.is_custom === true || key.includes('_csv_')) && 
                 dataset.language === selectedLanguage;
        });
        setCustomDatasets(customKeys);
      })
      .catch(err => {
        console.error('Error loading data:', err);
        setSttData({
          english: { 
            dataset: 'Error: Could not connect to backend. Please ensure the Flask server is running on port 5000.', 
            models: [] 
          },
          marathi: { dataset: 'Backend not connected', models: [] },
          hinglish: { dataset: 'Backend not connected', models: [] }
        });
      });
  }, [selectedLanguage]);

  // Load available models for testing
  const loadAvailableModels = () => {
    fetch('/api/available-models')
      .then(res => res.json())
      .then(data => {
        setAvailableTestModels(data.models || []);
      })
      .catch(err => {
        console.error('Error loading available models:', err);
        // Fallback to hardcoded list if API fails
        setAvailableTestModels([
          'Deepgram Nova 3', 'Deepgram Nova 2', 'Whisper', 
          'Google STT v2', 'AWS STT', 'AZURE STT', 
          'Gladia', 'AssemblyAI', 'Sarvam'
        ]);
      });
  };

  useEffect(() => {
    loadData();
    loadAvailableModels();
    // Poll for updates every 5 seconds
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, [refreshKey, selectedLanguage]);

  // Get current data based on selected dataset
  const currentData = useMemo(() => {
    if (selectedDataset === 'default') {
      return sttData[selectedLanguage] || { models: [] };
    } else {
      return sttData[selectedDataset] || { models: [] };
    }
  }, [sttData, selectedLanguage, selectedDataset]);

  const availableModels = currentData.models?.map(m => m.name) || [];

  const formatMetric = (value, isPercentage = false) => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'string') return value;
    if (isPercentage) return `${(value * 100).toFixed(2)}%`;
    return typeof value === 'number' ? value.toFixed(3) : value;
  };

  const getMetricColorClass = (value, metric) => {
    if (typeof value !== 'number') return 'neutral';
    
    if (metric.includes('wer') || metric.includes('cer')) {
      if (value < 0.1) return 'good';
      if (value < 0.2) return 'medium';
      return 'poor';
    }
    
    if (metric.includes('score')) {
      if (value > 4) return 'good';
      if (value > 3.5) return 'medium';
      return 'poor';
    }
    
    return 'neutral';
  };

  const getBestModel = (data, metric) => {
    if (!data || data.length === 0) return null;
    const validModels = data.filter(m => typeof m[metric] === 'number');
    if (validModels.length === 0) return null;
    
    if (metric.includes('wer') || metric.includes('cer')) {
      return validModels.reduce((best, current) => 
        current[metric] < best[metric] ? current : best
      );
    } else if (metric.includes('score')) {
      return validModels.reduce((best, current) => 
        current[metric] > best[metric] ? current : best
      );
    }
    return null;
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      setUploadedFile(file);
    } else {
      alert('Please upload a valid CSV file');
    }
  };

  const handleAudioUpload = (event) => {
    const file = event.target.files[0];
    if (file && (file.type.startsWith('audio/') || file.name.match(/\.(wav|mp3|m4a|flac)$/i))) {
      setUploadedAudioFile(file);
    } else {
      alert('Please upload a valid audio file');
    }
  };

  const handleModelToggle = (modelName) => {
    setSelectedTestModels(prev => {
      if (prev.includes(modelName)) {
        return prev.filter(m => m !== modelName);
      } else {
        return [...prev, modelName];
      }
    });
  };

  const startCsvTesting = async () => {
    if (!uploadedFile || !selectedTestModel) {
      alert('Please select a dataset and a model');
      return;
    }
    
    const formData = new FormData();
    
    // Handle default dataset for Hinglish
    if (uploadedFile === 'default-hinglish') {
      formData.append('use_default', 'true');
      formData.append('dataset_name', 'Default Hinglish Dataset');
    } else {
      formData.append('csv', uploadedFile);
      formData.append('dataset_name', `CSV: ${uploadedFile.name.replace('.csv', '')}`);
    }
    
    formData.append('model', selectedTestModel);
    formData.append('language', selectedLanguage);
    
    try {
      const response = await fetch('/api/test-csv', {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      if (result.task_id) {
        // Test started successfully - show notification and close modal
        const datasetName = uploadedFile === 'default-hinglish' ? 
          'Default Hinglish Dataset' : 
          uploadedFile.name;
        
        // Show success notification
        const notification = document.createElement('div');
        notification.style.cssText = `
          position: fixed;
          top: 20px;
          right: 20px;
          background: linear-gradient(135deg, #10b981 0%, #059669 100%);
          color: white;
          padding: 1rem 1.5rem;
          border-radius: 12px;
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
          z-index: 10000;
          animation: slideIn 0.3s ease-out;
          font-weight: 500;
        `;
        notification.innerHTML = `
          <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span style="font-size: 1.5rem;">✅</span>
            <div>
              <div style="font-weight: 600; margin-bottom: 0.25rem;">Test Started Successfully!</div>
              <div style="opacity: 0.9; font-size: 0.9rem;">
                Testing ${selectedTestModel} on ${datasetName}
              </div>
              <div style="opacity: 0.8; font-size: 0.85rem; margin-top: 0.25rem;">
                Results will appear in the All Models tab once completed
              </div>
            </div>
          </div>
        `;
        document.body.appendChild(notification);
        
        // Add animation
        const style = document.createElement('style');
        style.textContent = `
          @keyframes slideIn {
            from { transform: translateX(400px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
          }
        `;
        document.head.appendChild(style);
        
        // Remove notification after 5 seconds
        setTimeout(() => {
          notification.style.animation = 'slideOut 0.3s ease-in forwards';
          setTimeout(() => {
            document.body.removeChild(notification);
            document.head.removeChild(style);
          }, 300);
        }, 5000);
        
        // Add slide out animation
        const styleOut = document.createElement('style');
        styleOut.textContent = `
          @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(400px); opacity: 0; }
          }
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
          }
        `;
        document.head.appendChild(styleOut);
        
        // Reset form
        setUploadedFile(null);
        setSelectedTestModel('');
        
        // Optional: Poll for completion in background (without modal)
        const pollInterval = setInterval(async () => {
          const statusRes = await fetch(`/api/task/${result.task_id}`);
          const status = await statusRes.json();
          
          if (status.progress) {
            setTestingProgress(status.progress);
          }
          
          if (status.status === 'completed') {
            clearInterval(pollInterval);
            
            // Show completion notification
            const completionNotification = document.createElement('div');
            completionNotification.style.cssText = `
              position: fixed;
              top: 20px;
              right: 20px;
              background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
              color: white;
              padding: 1rem 1.5rem;
              border-radius: 12px;
              box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
              z-index: 10000;
              animation: slideIn 0.3s ease-out;
            `;
            completionNotification.innerHTML = `
              <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">🎉</span>
                <div>
                  <div style="font-weight: 600;">Testing Completed!</div>
                  <div style="opacity: 0.9; font-size: 0.9rem;">
                    Results have been added to the All Models tab
                  </div>
                </div>
              </div>
            `;
            document.body.appendChild(completionNotification);
            
            // Remove after 4 seconds
            setTimeout(() => {
              completionNotification.style.animation = 'slideOut 0.3s ease-in forwards';
              setTimeout(() => {
                document.body.removeChild(completionNotification);
              }, 300);
            }, 4000);
            
            // Reload data to include new dataset
            setRefreshKey(prev => prev + 1);
            
          } else if (status.status === 'failed') {
            clearInterval(pollInterval);
            
            // Show error notification
            const errorNotification = document.createElement('div');
            errorNotification.style.cssText = `
              position: fixed;
              top: 20px;
              right: 20px;
              background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
              color: white;
              padding: 1rem 1.5rem;
              border-radius: 12px;
              box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
              z-index: 10000;
              animation: slideIn 0.3s ease-out;
            `;
            errorNotification.innerHTML = `
              <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">❌</span>
                <div>
                  <div style="font-weight: 600;">Testing Failed</div>
                  <div style="opacity: 0.9; font-size: 0.9rem;">
                    ${status.error || 'An unexpected error occurred'}
                  </div>
                </div>
              </div>
            `;
            document.body.appendChild(errorNotification);
            
            // Remove after 5 seconds
            setTimeout(() => {
              errorNotification.style.animation = 'slideOut 0.3s ease-in forwards';
              setTimeout(() => {
                document.body.removeChild(errorNotification);
              }, 300);
            }, 5000);
          }
        }, 2000);
      }
    } catch (error) {
      console.error('Error:', error);
      
      // Show error notification
      const errorNotification = document.createElement('div');
      errorNotification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
      `;
      errorNotification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.75rem;">
          <span style="font-size: 1.5rem;">❌</span>
          <div>
            <div style="font-weight: 600;">Error Starting Test</div>
            <div style="opacity: 0.9; font-size: 0.9rem;">
              ${error.message || 'Failed to start the test. Please try again.'}
            </div>
          </div>
        </div>
      `;
      document.body.appendChild(errorNotification);
      
      // Remove after 5 seconds
      setTimeout(() => {
        errorNotification.style.animation = 'slideOut 0.3s ease-in forwards';
        setTimeout(() => {
          document.body.removeChild(errorNotification);
        }, 300);
      }, 5000);
    }
  };

  const startAudioTesting = async () => {
    if (!uploadedAudioFile || selectedTestModels.length === 0) {
      alert('Please upload an audio file and select at least one model');
      return;
    }
    
    setShowTestingModal(true);
    setTestingProgress(0);
    
    const formData = new FormData();
    formData.append('audio', uploadedAudioFile);
    selectedTestModels.forEach(model => {
      formData.append('models[]', model);
    });
    formData.append('language', selectedLanguage);
    
    try {
      const response = await fetch('/api/audio-test', {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      setTestingProgress(100);
      setAudioTestResults(result.results);
      
      setTimeout(() => {
        setShowTestingModal(false);
      }, 500);
    } catch (error) {
      console.error('Error:', error);
      alert('Error during audio testing');
      setShowTestingModal(false);
    }
  };

  const renderAudioTesting = () => (
    <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
      {/* Audio Testing Section */}
      <div style={{
        background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
        borderRadius: '16px',
        padding: '2rem',
        marginBottom: '2rem',
        color: 'white'
      }}>
        <h3 style={{ margin: '0 0 1rem', fontSize: '1.5rem' }}>🎙️ Audio File Testing</h3>
        <p style={{ fontSize: '1.1rem', opacity: 0.9 }}>
          Upload an audio file and compare transcription results across multiple models
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
        {/* Audio Upload */}
        <div style={{
          background: 'white',
          borderRadius: '12px',
          padding: '2rem',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        }}>
          <h4 style={{ margin: '0 0 1rem', color: '#1f2937' }}>📁 Upload Audio File</h4>
          <input
            type="file"
            accept="audio/*,.wav,.mp3,.m4a,.flac"
            onChange={handleAudioUpload}
            style={{ display: 'none' }}
            id="audio-upload"
          />
          <label
            htmlFor="audio-upload"
            style={{
              display: 'inline-block',
              padding: '0.75rem 1.5rem',
              background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
              color: 'white',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: '500'
            }}
          >
            Choose Audio File
          </label>
          {uploadedAudioFile && (
            <div style={{
              marginTop: '1rem',
              padding: '0.75rem',
              background: '#f0fdf4',
              borderRadius: '8px',
              border: '1px solid #bbf7d0'
            }}>
              <span style={{ color: '#16a34a' }}>✅ {uploadedAudioFile.name}</span>
            </div>
          )}
        </div>

        {/* Model Selection */}
        <div style={{
          background: 'white',
          borderRadius: '12px',
          padding: '2rem',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        }}>
          <h4 style={{ margin: '0 0 1rem', color: '#1f2937' }}>🤖 Select Models to Compare</h4>
          <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
            {(availableTestModels.length > 0 ? 
              availableTestModels.map(m => typeof m === 'string' ? m : m.display_name) : 
              ['Deepgram Nova 3', 'Deepgram Nova 2', 'Whisper', 'Google STT v2', 
               'AWS STT', 'AZURE STT', 'Gladia', 'AssemblyAI', 'Sarvam']
            ).map(model => (
              <label key={model} style={{
                display: 'flex',
                alignItems: 'center',
                padding: '0.5rem',
                cursor: 'pointer',
                borderRadius: '4px',
                transition: 'background 0.2s'
              }}>
                <input
                  type="checkbox"
                  checked={selectedTestModels.includes(model)}
                  onChange={() => handleModelToggle(model)}
                  style={{ marginRight: '0.5rem' }}
                />
                <span>{model}</span>
              </label>
            ))}
          </div>
        </div>
      </div>

      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <button
          onClick={startAudioTesting}
          disabled={!uploadedAudioFile || selectedTestModels.length === 0}
          style={{
            padding: '1rem 2rem',
            fontSize: '1.1rem',
            fontWeight: '600',
            color: 'white',
            background: (!uploadedAudioFile || selectedTestModels.length === 0)
              ? '#9ca3af'
              : 'linear-gradient(135deg, #10b981 0%, #047857 100%)',
            border: 'none',
            borderRadius: '12px',
            cursor: (!uploadedAudioFile || selectedTestModels.length === 0) ? 'not-allowed' : 'pointer'
          }}
        >
          🚀 Start Audio Comparison
        </button>
      </div>

      {/* Audio Test Results */}
      {audioTestResults && (
        <div style={{
          background: 'white',
          borderRadius: '12px',
          padding: '2rem',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        }}>
          <h3 style={{ margin: '0 0 1rem', color: '#1f2937' }}>📊 Transcription Results</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Model</th>
                  <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Transcription</th>
                  <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Processing Time</th>
                </tr>
              </thead>
              <tbody>
                {audioTestResults.map((result, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: '0.75rem', fontWeight: '500' }}>{result.model}</td>
                    <td style={{ padding: '0.75rem' }}>
                      {result.error ? (
                        <span style={{ color: '#dc2626' }}>Error: {result.error}</span>
                      ) : (
                        result.transcription
                      )}
                    </td>
                    <td style={{ padding: '0.75rem' }}>
                      {result.processingTime ? `${result.processingTime.toFixed(2)}s` : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );

  const renderCsvTesting = () => (
    <div style={{ maxWidth: '800px', margin: '0 auto' }}>
      {/* CSV Testing Section */}
      <div style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        borderRadius: '16px',
        padding: '2rem',
        marginBottom: '2rem',
        color: 'white'
      }}>
        <h3 style={{ margin: '0 0 1rem', fontSize: '1.5rem' }}>📋 CSV Batch Testing</h3>
        <p style={{ fontSize: '1.1rem', opacity: 0.9 }}>
          Upload a CSV file to test a model on multiple audio files. Results will be saved and appear as a new dataset in the All Models tab.
        </p>
      </div>

      <div style={{
        background: 'white',
        borderRadius: '12px',
        padding: '2rem',
        marginBottom: '1.5rem',
        border: '2px dashed #e2e8f0'
      }}>
        <h4 style={{ margin: '0 0 1rem', color: '#1f2937' }}>📄 Select Dataset</h4>
        
        {/* Add Default Dataset option for Hinglish */}
        {selectedLanguage === 'hinglish' && (
          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{
              display: 'flex',
              alignItems: 'center',
              padding: '0.75rem',
              background: uploadedFile === 'default-hinglish' ? '#dbeafe' : '#f3f4f6',
              borderRadius: '8px',
              cursor: 'pointer',
              marginBottom: '0.5rem',
              border: uploadedFile === 'default-hinglish' ? '2px solid #3b82f6' : '2px solid transparent'
            }}>
              <input
                type="radio"
                name="datasetSource"
                value="default"
                checked={uploadedFile === 'default-hinglish'}
                onChange={() => setUploadedFile('default-hinglish')}
                style={{ marginRight: '0.75rem' }}
              />
              <div>
                <span style={{ fontWeight: '600', color: '#1f2937' }}>Use Default Hinglish Dataset</span>
                <p style={{ margin: '0.25rem 0 0', fontSize: '0.875rem', color: '#6b7280' }}>
                  Pre-loaded dataset for testing Hinglish models
                </p>
              </div>
            </label>
            <label style={{
              display: 'flex',
              alignItems: 'center',
              padding: '0.75rem',
              background: uploadedFile && uploadedFile !== 'default-hinglish' ? '#dbeafe' : '#f3f4f6',
              borderRadius: '8px',
              cursor: 'pointer',
              border: uploadedFile && uploadedFile !== 'default-hinglish' ? '2px solid #3b82f6' : '2px solid transparent'
            }}>
              <input
                type="radio"
                name="datasetSource"
                value="custom"
                checked={uploadedFile && uploadedFile !== 'default-hinglish'}
                onChange={() => setUploadedFile(null)}
                style={{ marginRight: '0.75rem' }}
              />
              <div>
                <span style={{ fontWeight: '600', color: '#1f2937' }}>Upload Custom CSV</span>
                <p style={{ margin: '0.25rem 0 0', fontSize: '0.875rem', color: '#6b7280' }}>
                  Upload your own CSV file for testing
                </p>
              </div>
            </label>
          </div>
        )}
        
        {/* Show file upload section */}
        {(selectedLanguage !== 'hinglish' || uploadedFile !== 'default-hinglish') && (
          <>
            <p style={{ color: '#6b7280', marginBottom: '1rem' }}>
              CSV should have columns: Key/audio_path and Transcription/ground_truth
            </p>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              style={{ display: 'none' }}
              id="csv-upload"
            />
            <label
              htmlFor="csv-upload"
              style={{
                display: 'inline-block',
                padding: '0.75rem 1.5rem',
                background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
                color: 'white',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: '500'
              }}
            >
              Choose CSV File
            </label>
            {uploadedFile && uploadedFile !== 'default-hinglish' && (
              <div style={{
                marginTop: '1rem',
                padding: '0.75rem',
                background: '#f0fdf4',
                borderRadius: '8px',
                border: '1px solid #bbf7d0'
              }}>
                <span style={{ color: '#16a34a' }}>✅ {uploadedFile.name}</span>
              </div>
            )}
          </>
        )}
        
        {/* Show confirmation for default dataset */}
        {uploadedFile === 'default-hinglish' && (
          <div style={{
            marginTop: '1rem',
            padding: '0.75rem',
            background: '#f0fdf4',
            borderRadius: '8px',
            border: '1px solid #bbf7d0'
          }}>
            <span style={{ color: '#16a34a' }}>✅ Default Hinglish Dataset Selected</span>
          </div>
        )}
      </div>

      <div style={{
        background: 'white',
        borderRadius: '12px',
        padding: '2rem',
        marginBottom: '2rem',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
      }}>
        <h4 style={{ margin: '0 0 1rem', color: '#1f2937' }}>🤖 Select Model for Testing</h4>
        <select
          value={selectedTestModel}
          onChange={(e) => setSelectedTestModel(e.target.value)}
          style={{
            width: '100%',
            padding: '0.75rem',
            borderRadius: '8px',
            border: '2px solid #e5e7eb',
            fontSize: '1rem'
          }}
        >
          <option value="">Choose a model to test...</option>
          {(availableTestModels.length > 0 ? 
            availableTestModels.map(m => typeof m === 'string' ? m : m.display_name) : 
            ['Deepgram Nova 3', 'Deepgram Nova 2', 'Whisper', 'Google STT v2', 
             'AWS STT', 'AZURE STT', 'Gladia', 'AssemblyAI', 'Sarvam']
          ).map(model => (
            <option key={model} value={model}>{model}</option>
          ))}
        </select>
      </div>

      <div style={{ textAlign: 'center' }}>
        <button
          onClick={startCsvTesting}
          disabled={!uploadedFile || !selectedTestModel}
          style={{
            padding: '1rem 2rem',
            fontSize: '1.1rem',
            fontWeight: '600',
            color: 'white',
            background: (!uploadedFile || !selectedTestModel)
              ? '#9ca3af'
              : 'linear-gradient(135deg, #10b981 0%, #047857 100%)',
            border: 'none',
            borderRadius: '12px',
            cursor: (!uploadedFile || !selectedTestModel) ? 'not-allowed' : 'pointer'
          }}
        >
          🚀 Start Batch Testing
        </button>
      </div>

      {/* Custom Datasets List */}
      {customDatasets.length > 0 && (
        <div style={{
          marginTop: '2rem',
          background: 'white',
          borderRadius: '12px',
          padding: '1.5rem',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        }}>
          <h4 style={{ margin: '0 0 1rem', color: '#1f2937' }}>📂 Processed CSV Datasets for {selectedLanguage}</h4>
          <ul style={{ margin: 0, paddingLeft: '1.5rem' }}>
            {customDatasets.map((datasetKey, idx) => {
              const dataset = sttData[datasetKey];
              return (
                <li key={idx} style={{ color: '#6b7280', marginBottom: '0.5rem' }}>
                  {dataset?.dataset || datasetKey} 
                  {dataset?.models?.length > 0 && ` - ${dataset.models.length} model(s) tested`}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );

  const renderOverview = () => (
    <div>
      {/* Dataset Selection for custom datasets */}
      {customDatasets.length > 0 && (
        <div style={{ marginBottom: '2rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#374151' }}>
            📂 Select Dataset
          </label>
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            style={{
              padding: '0.75rem',
              borderRadius: '8px',
              border: '2px solid #e5e7eb',
              fontSize: '1rem',
              background: 'white',
              minWidth: '300px'
            }}
          >
            <option value="default">Default Dataset</option>
            {customDatasets.map(datasetKey => {
              const dataset = sttData[datasetKey];
              return (
                <option key={datasetKey} value={datasetKey}>
                  {dataset?.dataset || datasetKey}
                </option>
              );
            })}
          </select>
        </div>
      )}

      {/* Dataset Info Card */}
      <div style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        borderRadius: '16px',
        padding: '2rem',
        marginBottom: '2rem',
        color: 'white'
      }}>
        <h3 style={{ margin: '0 0 0.5rem', fontSize: '1.5rem' }}>📊 Dataset Information</h3>
        <p style={{ fontSize: '1.1rem', marginBottom: '1rem' }}>
          {currentData.dataset || 'Loading dataset information...'}
        </p>
        <p style={{ fontSize: '1rem', fontWeight: 'bold' }}>
          🔍 Models evaluated: <span style={{ fontSize: '1.5rem' }}>{currentData.models?.length || 0}</span>
        </p>
        {currentData.is_custom && (
          <p style={{ fontSize: '0.9rem', marginTop: '0.5rem', opacity: 0.9 }}>
            💾 Custom CSV Dataset - Language: {currentData.language}
          </p>
        )}
      </div>

      {/* Quick Stats */}
      {currentData.models && currentData.models.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
          {(selectedLanguage === 'english' || currentData.language === 'english') && (
            <>
              <div style={{
                background: 'linear-gradient(135deg, #10b981 0%, #047857 100%)',
                borderRadius: '12px',
                padding: '1.5rem',
                color: 'white'
              }}>
                <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.9rem', opacity: 0.9 }}>🏆 Best WER (Clean)</h4>
                <p style={{ margin: '0 0 0.25rem', fontSize: '1.5rem', fontWeight: 'bold' }}>
                  {formatMetric(getBestModel(currentData.models, 'wer_clean')?.wer_clean, true)}
                </p>
                <p style={{ margin: 0, fontSize: '0.8rem', opacity: 0.8 }}>{getBestModel(currentData.models, 'wer_clean')?.name}</p>
              </div>
              <div style={{
                background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
                borderRadius: '12px',
                padding: '1.5rem',
                color: 'white'
              }}>
                <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.9rem', opacity: 0.9 }}>⚡ Fastest Streaming</h4>
                <p style={{ margin: '0 0 0.25rem', fontSize: '1.5rem', fontWeight: 'bold' }}>
                  {currentData.models?.filter(m => typeof m.latency_streaming === 'number' && m.latency_streaming > 0).length > 0
                    ? `${Math.min(...currentData.models.filter(m => typeof m.latency_streaming === 'number' && m.latency_streaming > 0).map(m => m.latency_streaming)).toFixed(3)}s`
                    : 'N/A'}
                </p>
                <p style={{ margin: 0, fontSize: '0.8rem', opacity: 0.8 }}>Latency</p>
              </div>
            </>
          )}
          {(selectedLanguage === 'hinglish' || currentData.language === 'hinglish') && (
            <div style={{
              background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
              borderRadius: '12px',
              padding: '1.5rem',
              color: 'white'
            }}>
              <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.9rem', opacity: 0.9 }}>⭐ Best Score</h4>
              <p style={{ margin: '0 0 0.25rem', fontSize: '1.5rem', fontWeight: 'bold' }}>
                {formatMetric(getBestModel(currentData.models, 'score')?.score)}/5.0
              </p>
              <p style={{ margin: 0, fontSize: '0.8rem', opacity: 0.8 }}>{getBestModel(currentData.models, 'score')?.name}</p>
            </div>
          )}
          {(selectedLanguage === 'marathi' || currentData.language === 'marathi') && (
            <>
              <div style={{
                background: 'linear-gradient(135deg, #10b981 0%, #047857 100%)',
                borderRadius: '12px',
                padding: '1.5rem',
                color: 'white'
              }}>
                <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.9rem', opacity: 0.9 }}>🏆 Best WER</h4>
                <p style={{ margin: '0 0 0.25rem', fontSize: '1.5rem', fontWeight: 'bold' }}>
                  {formatMetric(getBestModel(currentData.models, 'wer')?.wer, true)}
                </p>
                <p style={{ margin: 0, fontSize: '0.8rem', opacity: 0.8 }}>{getBestModel(currentData.models, 'wer')?.name}</p>
              </div>
              <div style={{
                background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
                borderRadius: '12px',
                padding: '1.5rem',
                color: 'white'
              }}>
                <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.9rem', opacity: 0.9 }}>🎯 Best CER</h4>
                <p style={{ margin: '0 0 0.25rem', fontSize: '1.5rem', fontWeight: 'bold' }}>
                  {formatMetric(getBestModel(currentData.models, 'cer')?.cer, true)}
                </p>
                <p style={{ margin: 0, fontSize: '0.8rem', opacity: 0.8 }}>{getBestModel(currentData.models, 'cer')?.name}</p>
              </div>
            </>
          )}
        </div>
      )}

      {/* Performance Chart */}
      {(selectedLanguage === 'english' || currentData.language === 'english') && currentData.models?.length > 0 && currentData.models.some(m => m.wer_clean) && (
        <div style={{
          background: 'white',
          borderRadius: '12px',
          padding: '1.5rem',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        }}>
          <h3 style={{ margin: '0 0 1rem', color: '#1f2937' }}>📝 Word Error Rate Comparison (Clean)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={currentData.models.filter(m => m.wer_clean)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} tick={{fontSize: 12}} />
              <YAxis tick={{fontSize: 12}} />
              <Tooltip 
                formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'WER']} 
                contentStyle={{
                  backgroundColor: '#f8fafc',
                  border: '1px solid #e2e8f0',
                  borderRadius: '12px'
                }}
              />
              <Bar dataKey="wer_clean" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* No Data Message */}
      {(!currentData.models || currentData.models.length === 0) && (
        <div style={{
          background: 'white',
          borderRadius: '12px',
          padding: '3rem',
          textAlign: 'center',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        }}>
          <p style={{ color: '#6b7280', fontSize: '1.1rem' }}>
            {currentData.dataset?.includes('Error') ? 
              '⚠️ Cannot connect to backend server. Please ensure Flask is running on port 5000.' :
              '📊 No model data available for this dataset.'}
          </p>
        </div>
      )}
    </div>
  );

  const renderComparison = () => {
    if (!selectedModel1 || !selectedModel2) {
      return (
        <div style={{
          textAlign: 'center',
          padding: '4rem 2rem',
          color: '#6b7280'
        }}>
          <h3 style={{ margin: '0 0 0.5rem', color: '#374151' }}>Ready to Compare Models?</h3>
          <p style={{ margin: 0 }}>Select two models from the dropdowns above to see detailed comparison</p>
        </div>
      );
    }

    const model1 = currentData.models?.find(m => m.name === selectedModel1);
    const model2 = currentData.models?.find(m => m.name === selectedModel2);
    
    if (!model1 || !model2) return null;
    
    const metrics = Object.keys(model1).filter(key => key !== 'name' && key !== 'dataset_source');

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        <div style={{
          background: 'white',
          borderRadius: '12px',
          padding: '1.5rem',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
          border: '2px solid #3b82f6'
        }}>
          <h3 style={{ margin: '0 0 1rem', color: '#3b82f6' }}>{selectedModel1}</h3>
          {metrics.map(metric => (
            <div key={metric} style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '0.5rem 0',
              borderBottom: '1px solid #f3f4f6'
            }}>
              <span style={{ color: '#6b7280', textTransform: 'capitalize' }}>{metric.replace(/_/g, ' ')}</span>
              <span style={{ fontWeight: 'bold' }}>
                {formatMetric(model1[metric], metric.includes('wer') || metric.includes('cer'))}
              </span>
            </div>
          ))}
        </div>

        <div style={{
          background: 'white',
          borderRadius: '12px',
          padding: '1.5rem',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
          border: '2px solid #10b981'
        }}>
          <h3 style={{ margin: '0 0 1rem', color: '#10b981' }}>{selectedModel2}</h3>
          {metrics.map(metric => (
            <div key={metric} style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '0.5rem 0',
              borderBottom: '1px solid #f3f4f6'
            }}>
              <span style={{ color: '#6b7280', textTransform: 'capitalize' }}>{metric.replace(/_/g, ' ')}</span>
              <span style={{ fontWeight: 'bold' }}>
                {formatMetric(model2[metric], metric.includes('wer') || metric.includes('cer'))}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderAllModels = () => {
    // Determine which language this dataset is for
    const datasetLanguage = currentData.language || selectedLanguage;
    
    return (
      <div>
        {/* Dataset Selector */}
        <div style={{ marginBottom: '2rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#374151' }}>
            📂 Select Dataset to View
          </label>
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            style={{
              padding: '0.75rem',
              borderRadius: '8px',
              border: '2px solid #e5e7eb',
              fontSize: '1rem',
              background: 'white',
              minWidth: '300px'
            }}
          >
            <optgroup label="Standard Datasets">
              <option value="default">Default ({selectedLanguage})</option>
            </optgroup>
            {customDatasets.length > 0 && (
              <optgroup label={`Custom CSV Datasets (${selectedLanguage})`}>
                {customDatasets.map(datasetKey => {
                  const dataset = sttData[datasetKey];
                  return (
                    <option key={datasetKey} value={datasetKey}>
                      {dataset?.dataset || datasetKey}
                    </option>
                  );
                })}
              </optgroup>
            )}
          </select>
        </div>

        <div style={{
          background: 'white',
          borderRadius: '12px',
          padding: '1.5rem',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        }}>
          <h3 style={{ margin: '0 0 1rem' }}>
            📊 Complete Models Comparison - {currentData.dataset || 'Dataset'}
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>🤖 Model</th>
                  {(datasetLanguage === 'english') && (
                    <>
                      <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>📝 WER Clean</th>
                      <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>🔤 CER Clean</th>
                    </>
                  )}
                  {(datasetLanguage === 'marathi') && (
                    <>
                      <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>📊 WER</th>
                      <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>🎯 CER</th>
                    </>
                  )}
                  {(datasetLanguage === 'hinglish') && (
                    <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>⭐ Score</th>
                  )}
                  <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>⏱️ Batch Latency</th>
                  <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>📡 Stream Latency</th>
                  <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>💰 Cost</th>
                  <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>🔴 Streaming</th>
                  {currentData.models?.some(m => m.dataset_source) && (
                    <th style={{ padding: '0.75rem', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>📂 Source</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {currentData.models?.map((model, index) => (
                  <tr key={index} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: '0.75rem', fontWeight: '500' }}>{model.name}</td>
                    {(datasetLanguage === 'english') && (
                      <>
                        <td style={{ padding: '0.75rem' }}>
                          <span style={{
                            color: getMetricColorClass(model.wer_clean, 'wer') === 'good' ? '#059669' : 
                                   getMetricColorClass(model.wer_clean, 'wer') === 'medium' ? '#d97706' : '#dc2626'
                          }}>
                            {formatMetric(model.wer_clean, true)}
                          </span>
                        </td>
                        <td style={{ padding: '0.75rem' }}>
                          <span style={{
                            color: getMetricColorClass(model.cer_clean, 'cer') === 'good' ? '#059669' : 
                                   getMetricColorClass(model.cer_clean, 'cer') === 'medium' ? '#d97706' : '#dc2626'
                          }}>
                            {formatMetric(model.cer_clean, true)}
                          </span>
                        </td>
                      </>
                    )}
                    {(datasetLanguage === 'marathi') && (
                      <>
                        <td style={{ padding: '0.75rem' }}>
                          <span style={{
                            color: getMetricColorClass(model.wer, 'wer') === 'good' ? '#059669' : 
                                   getMetricColorClass(model.wer, 'wer') === 'medium' ? '#d97706' : '#dc2626'
                          }}>
                            {formatMetric(model.wer, true)}
                          </span>
                        </td>
                        <td style={{ padding: '0.75rem' }}>
                          <span style={{
                            color: getMetricColorClass(model.cer, 'cer') === 'good' ? '#059669' : 
                                   getMetricColorClass(model.cer, 'cer') === 'medium' ? '#d97706' : '#dc2626'
                          }}>
                            {formatMetric(model.cer, true)}
                          </span>
                        </td>
                      </>
                    )}
                    {(datasetLanguage === 'hinglish') && (
                      <td style={{ padding: '0.75rem' }}>
                        <span style={{
                          color: getMetricColorClass(model.score, 'score') === 'good' ? '#059669' : 
                                 getMetricColorClass(model.score, 'score') === 'medium' ? '#d97706' : '#dc2626'
                        }}>
                          {formatMetric(model.score)}/5.0
                        </span>
                      </td>
                    )}
                    <td style={{ padding: '0.75rem' }}>
                      {typeof model.latency_batch === 'number' ? `${model.latency_batch}s` : model.latency_batch || 'N/A'}
                    </td>
                    <td style={{ padding: '0.75rem' }}>
                      {typeof model.latency_streaming === 'number' ? `${model.latency_streaming}s` : model.latency_streaming || 'N/A'}
                    </td>
                    <td style={{ padding: '0.75rem' }}>
                      {typeof model.cost_batch === 'number' ? `$${model.cost_batch}` : model.cost_batch || 'N/A'}
                    </td>
                    <td style={{ padding: '0.75rem' }}>
                      <span style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.25rem',
                        padding: '0.25rem 0.5rem',
                        borderRadius: '6px',
                        fontSize: '0.875rem',
                        fontWeight: '500',
                        background: model.streaming ? '#dcfce7' : '#fee2e2',
                        color: model.streaming ? '#166534' : '#991b1b'
                      }}>
                        {model.streaming ? '✅ Yes' : '❌ No'}
                      </span>
                    </td>
                    {currentData.models?.some(m => m.dataset_source) && (
                      <td style={{ padding: '0.75rem', fontSize: '0.875rem', color: '#6b7280' }}>
                        {model.dataset_source || 'Default'}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)', padding: '2rem' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{
          textAlign: 'center',
          marginBottom: '2rem',
          background: 'white',
          borderRadius: '16px',
          padding: '2rem',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        }}>
          <h1 style={{ margin: '0 0 0.5rem', color: '#1f2937', fontSize: '2.25rem' }}>🎙️Model Comparison</h1>
          <p style={{ margin: 0, color: '#6b7280', fontSize: '1.1rem' }}>
            Compare performance metrics between different Speech-to-Text models across languages
          </p>
        </div>

        {/* Language Selection */}
        <div style={{ marginBottom: '2rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#374151' }}>
            🌍 Select Language
          </label>
          <select
            value={selectedLanguage}
            onChange={(e) => {
              setSelectedLanguage(e.target.value);
              setSelectedDataset('default');
            }}
            style={{
              padding: '0.75rem',
              borderRadius: '8px',
              border: '2px solid #e5e7eb',
              fontSize: '1rem',
              background: 'white',
              minWidth: '200px'
            }}
          >
            <option value="english">🇺🇸 English</option>
            <option value="hinglish">🇮🇳 Hinglish</option>
            <option value="marathi">🇮🇳 Marathi</option>
          </select>
        </div>

        {/* Navigation Tabs */}
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
          {['overview', 'comparison', 'all-models', 'audio-test', 'csv-test'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: '0.75rem 1.5rem',
                borderRadius: '8px',
                border: 'none',
                fontSize: '1rem',
                fontWeight: '500',
                cursor: 'pointer',
                background: activeTab === tab 
                  ? 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)' 
                  : 'white',
                color: activeTab === tab ? 'white' : '#6b7280',
                boxShadow: activeTab === tab 
                  ? '0 4px 12px rgba(59, 130, 246, 0.3)' 
                  : '0 2px 4px rgba(0, 0, 0, 0.1)',
                transition: 'all 0.3s ease'
              }}
            >
              {tab === 'overview' && '📊 '}
              {tab === 'comparison' && '⚖️ '}
              {tab === 'all-models' && '📋 '}
              {tab === 'audio-test' && '🎙️ '}
              {tab === 'csv-test' && '📂 '}
              {tab.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
            </button>
          ))}
        </div>

        {/* Model Comparison Dropdowns */}
        {activeTab === 'comparison' && (
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#374151' }}>
                🤖 Model 1
              </label>
              <select
                value={selectedModel1}
                onChange={(e) => setSelectedModel1(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  borderRadius: '8px',
                  border: '2px solid #3b82f6',
                  fontSize: '1rem',
                  background: 'white'
                }}
              >
                <option value="">Select a model</option>
                {availableModels.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#374151' }}>
                🤖 Model 2
              </label>
              <select
                value={selectedModel2}
                onChange={(e) => setSelectedModel2(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  borderRadius: '8px',
                  border: '2px solid #10b981',
                  fontSize: '1rem',
                  background: 'white'
                }}
              >
                <option value="">Select a model</option>
                {availableModels.filter(model => model !== selectedModel1).map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>
          </div>
        )}

        {/* Content */}
        <div>
          {activeTab === 'overview' && renderOverview()}
          {activeTab === 'comparison' && renderComparison()}
          {activeTab === 'all-models' && renderAllModels()}
          {activeTab === 'audio-test' && renderAudioTesting()}
          {activeTab === 'csv-test' && renderCsvTesting()}
        </div>
      </div>

      {/* Testing Modal */}
      {showTestingModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: 'white',
            borderRadius: '16px',
            padding: '2rem',
            maxWidth: '400px',
            width: '90%',
            textAlign: 'center',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.2)'
          }}>
            <h3 style={{ margin: '0 0 0.5rem', color: '#1f2937' }}>🧪 Testing in Progress</h3>
            <p style={{ margin: '0 0 1.5rem', color: '#6b7280' }}>
              Running {selectedLanguage} pipeline...
            </p>
            
            <div style={{
              width: '100%',
              height: '8px',
              background: '#f3f4f6',
              borderRadius: '4px',
              overflow: 'hidden',
              marginBottom: '1rem'
            }}>
              <div style={{
                width: '100%',
                height: '100%',
                background: 'linear-gradient(135deg, #10b981 0%, #047857 100%)',
                animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite'
              }} />
            </div>
            
            <p style={{ 
              margin: '0 0 1rem', 
              color: '#374151', 
              fontWeight: '500',
              fontSize: '1rem',
              lineHeight: '1.5'
            }}>
              This will take 2-10 minutes to complete based on the audio length.
            </p>
            
            <p style={{ 
              margin: '0', 
              color: '#dc2626', 
              fontWeight: '600',
              fontSize: '0.9rem',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              DO NOT REFRESH
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default STTDashboard;
