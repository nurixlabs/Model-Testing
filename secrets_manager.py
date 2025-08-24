#!/usr/bin/env python3
"""
AWS Secrets Manager utility for retrieving application secrets
"""
import json
import logging
import os
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

class SecretsManager:
    """AWS Secrets Manager client for retrieving application secrets"""
    
    def __init__(self, region_name: str = "ap-south-1"):
        """
        Initialize Secrets Manager client
        
        Args:
            region_name: AWS region where secrets are stored
        """
        self.region_name = region_name
        self.secret_arn = "arn:aws:secretsmanager:ap-south-1:533266975263:secret:prod/ASR-Dashboard-SD8xuA"
        self._client = None
        self._secrets_cache = {}
        
    @property
    def client(self):
        if self._client is None:
            try:
                akid = os.environ.get('AKID')
                skey = os.environ.get('SKEY')
                print("akid>>>>>>>>>>>",akid)

                if akid and skey:
                    self._client = boto3.client(
                        "secretsmanager",
                        region_name=self.region_name,
                        aws_access_key_id=akid,
                        aws_secret_access_key=skey,
                        # aws_session_token=stok,
                    )
                else:
                    # fallback to normal boto3 chain (useful if you later move to IRSA)
                    self._client = boto3.client("secretsmanager", region_name=self.region_name)
            except NoCredentialsError:
                raise RuntimeError("❌ AWS static credentials not found in environment")
        return self._client
    
    def get_secrets(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Retrieve all secrets from AWS Secrets Manager
        
        Args:
            force_refresh: Force refresh of cached secrets
            
        Returns:
            Dictionary containing all secrets
        """
        if not force_refresh and self._secrets_cache:
            return self._secrets_cache
            
        try:
            logger.info(f"Retrieving secrets from: {self.secret_arn}")
            response = self.client.get_secret_value(SecretId=self.secret_arn)
            
            secret_string = response.get('SecretString')
            if secret_string:
                secrets = json.loads(secret_string)
                self._secrets_cache = secrets
                logger.info("Successfully retrieved secrets from AWS Secrets Manager")
                return secrets
            else:
                logger.error("Secret string is empty")
                return {}
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                logger.error(f"Secret not found: {self.secret_arn}")
            elif error_code == 'InvalidRequestException':
                logger.error(f"Invalid request for secret: {self.secret_arn}")
            elif error_code == 'InvalidParameterException':
                logger.error(f"Invalid parameter for secret: {self.secret_arn}")
            elif error_code == 'DecryptionFailure':
                logger.error(f"Failed to decrypt secret: {self.secret_arn}")
            elif error_code == 'AccessDeniedException':
                logger.error(f"Access denied to secret: {self.secret_arn}")
            else:
                logger.error(f"Unexpected error retrieving secret: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse secret JSON: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {}
    
    def get_secret_value(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a specific secret value by key
        
        Args:
            key: Secret key to retrieve
            default: Default value if key not found
            
        Returns:
            Secret value or default
        """
        secrets = self.get_secrets()
        return secrets.get(key, default)
    
    def load_secrets_to_env(self, force_refresh: bool = False) -> bool:
        """
        Load all secrets into environment variables
        
        Args:
            force_refresh: Force refresh of cached secrets
            
        Returns:
            True if secrets were loaded successfully, False otherwise
        """
        try:
            secrets = self.get_secrets(force_refresh)
            if not secrets:
                logger.warning("No secrets retrieved from AWS Secrets Manager")
                return False
                
            # Load secrets into environment
            for key, value in secrets.items():
                if isinstance(value, str):
                    os.environ[key] = value
                    logger.debug(f"Loaded secret: {key}")
                else:
                    logger.warning(f"Skipping non-string secret: {key}")
            
            logger.info(f"Successfully loaded {len(secrets)} secrets into environment")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load secrets to environment: {e}")
            return False


def initialize_secrets() -> bool:
    """
    Initialize secrets from AWS Secrets Manager
    This function should be called early in the application startup
    
    Returns:
        True if secrets were loaded successfully, False otherwise
    """
    try:
        # Check if we're running in Kubernetes with IRSA
        if os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount/token'):
            logger.info("Running in Kubernetes with service account, using IRSA for AWS authentication")
        else:
            logger.info("Not running in Kubernetes, using default AWS credentials")
        
        secrets_manager = SecretsManager()
        return secrets_manager.load_secrets_to_env()
        
    except Exception as e:
        logger.error(f"Failed to initialize secrets: {e}")
        return False


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to get a secret value
    First tries environment variable, then AWS Secrets Manager
    
    Args:
        key: Secret key to retrieve
        default: Default value if key not found
        
    Returns:
        Secret value or default
    """
    # First try environment variable
    env_value = os.environ.get(key)
    if env_value:
        print("PRINTING ENV VALUE", env_value)
        return env_value
    
    # Then try Secrets Manager
    try:
        secrets_manager = SecretsManager()
        return secrets_manager.get_secret_value(key, default)
    except Exception as e:
        logger.warning(f"Failed to retrieve secret {key} from Secrets Manager: {e}")
        return default


if __name__ == "__main__":
    # Test the secrets manager
    logging.basicConfig(level=logging.INFO)
    
    print("Testing AWS Secrets Manager integration...")
    
    if initialize_secrets():
        print("✅ Secrets loaded successfully")
        
        # Test retrieving a specific secret
        api_key = get_secret('DEEPGRAM_API_KEY')
        if api_key:
            print(f"✅ DEEPGRAM_API_KEY found (length: {len(api_key)})")
        else:
            print("❌ DEEPGRAM_API_KEY not found")
    else:
        print("❌ Failed to load secrets")
