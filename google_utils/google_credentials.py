"""
Utility for handling Google Cloud credentials from environment variables.
"""
import os
import json
import tempfile
import logging
from typing import Optional, Dict, Any
from google.oauth2 import service_account
from google.auth.credentials import Credentials

logger = logging.getLogger(__name__)


def get_google_credentials() -> Optional[Credentials]:
    """
    Get Google Cloud credentials from environment variable.
    
    Tries the following in order:
    1. GOOGLE_APPLICATION_CREDENTIALS_JSON - JSON string containing service account
    2. GOOGLE_APPLICATION_CREDENTIALS - Path to service account file
    
    Returns:
        Google credentials object or None if not found
    """
    # Try JSON string first
    creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if creds_json:
        try:
            # Remove surrounding quotes if present
            if creds_json.startswith("'") and creds_json.endswith("'"):
                creds_json = creds_json[1:-1]
            elif creds_json.startswith('"') and creds_json.endswith('"'):
                creds_json = creds_json[1:-1]
                
            # Parse JSON
            creds_data = json.loads(creds_json)
            
            # Create credentials from the JSON data
            credentials = service_account.Credentials.from_service_account_info(
                creds_data,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            
            logger.info("Google credentials loaded from GOOGLE_APPLICATION_CREDENTIALS_JSON")
            return credentials
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
        except Exception as e:
            logger.error(f"Error creating credentials from JSON: {e}")
    
    # Try file path
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if creds_path and os.path.exists(creds_path):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            logger.info(f"Google credentials loaded from file: {creds_path}")
            return credentials
        except Exception as e:
            logger.error(f"Error loading credentials from file: {e}")
    
    logger.warning("No Google credentials found in environment")
    return None


def get_credentials_info() -> Optional[Dict[str, Any]]:
    """
    Get Google Cloud credentials info as a dictionary.
    
    Returns:
        Dictionary containing service account info or None if not found
    """
    # Try JSON string first
    creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if creds_json:
        try:
            # Remove surrounding quotes if present
            if creds_json.startswith("'") and creds_json.endswith("'"):
                creds_json = creds_json[1:-1]
            elif creds_json.startswith('"') and creds_json.endswith('"'):
                creds_json = creds_json[1:-1]
                
            # Parse and return JSON
            return json.loads(creds_json)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
    
    # Try file path
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if creds_path and os.path.exists(creds_path):
        try:
            with open(creds_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading credentials from file: {e}")
    
    return None


def create_temp_credentials_file() -> Optional[str]:
    """
    Create a temporary credentials file from environment variable.
    
    This is useful for tools that require a file path instead of credentials object.
    
    Returns:
        Path to temporary credentials file or None if credentials not found
    """
    creds_info = get_credentials_info()
    if not creds_info:
        return None
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(creds_info, f, indent=2)
            temp_path = f.name
        
        logger.info(f"Created temporary credentials file: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Error creating temporary credentials file: {e}")
        return None


def setup_google_environment():
    """
    Set up Google Cloud environment from embedded credentials.
    
    This creates a temporary file and sets GOOGLE_APPLICATION_CREDENTIALS
    if only GOOGLE_APPLICATION_CREDENTIALS_JSON is present.
    """
    # Only proceed if JSON is present but file path is not
    if (os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON') and 
        not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')):
        
        temp_path = create_temp_credentials_file()
        if temp_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_path
            logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS to temporary file: {temp_path}")