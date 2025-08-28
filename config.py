#!/usr/bin/env python3
"""
Configuration Module - AI Model Response Evaluator

Centralized configuration management for environment variables, logging,
application constants, and feature flags.

Author: dvirla
License: MIT
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# APPLICATION CONSTANTS
# =============================================================================

# Database configuration
DATABASE_PATH = 'new_query_history.db'

# Default evaluation prompt template
DEFAULT_EVALUATION_PROMPT = """You are an expert AI evaluator. Please analyze the following AI responses and provide a qualitative evaluation.

Original Prompt: {original_prompt}

Gemini WITHOUT Search:
Thinking: {gemini_without_search_thinking}
Response: {gemini_without_search_response}

Gemini WITH Search:
Thinking: {gemini_with_search_thinking}  
Response: {gemini_with_search_response}

Please evaluate considering:
1. Completeness and thoroughness
2. Clarity and coherence  
3. Reasoning quality
4. Relevance to the original prompt
5. Differences between the two approaches

Provide your evaluation in the following format:
SCORE: [1-5 scale where 5 is excellent]
REASONING: [Your detailed analysis]
STRENGTHS: [What the responses did well]
WEAKNESSES: [Areas for improvement]
COMPARISON: [How the two approaches differ]"""

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

class Config:
    """Centralized configuration class with environment variable management."""
    
    # Required API Keys
    GOOGLE_API_KEY: Optional[str] = os.getenv('GOOGLE_API_KEY')
    TAVILY_API_KEY: Optional[str] = os.getenv('TAVILY_API_KEY')
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
    OLLAMA_KEEP_ALIVE: str = os.getenv('OLLAMA_KEEP_ALIVE', '5m')
    OLLAMA_MAX_LOADED_MODELS: int = int(os.getenv('OLLAMA_MAX_LOADED_MODELS', '3'))
    OLLAMA_NUM_PARALLEL: int = int(os.getenv('OLLAMA_NUM_PARALLEL', '2'))
    
    # Flask Application Settings
    FLASK_ENV: str = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG: bool = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    FLASK_HOST: str = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT: int = int(os.getenv('FLASK_PORT', '5001'))
    FLASK_SECRET_KEY: str = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')
    
    # Database Configuration
    DATABASE_ECHO: bool = os.getenv('DATABASE_ECHO', 'False').lower() == 'true'
    DATABASE_POOL_SIZE: int = int(os.getenv('DATABASE_POOL_SIZE', '10'))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: Optional[str] = os.getenv('LOG_FILE')
    LOG_MAX_SIZE: int = int(os.getenv('LOG_MAX_SIZE', '10485760'))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    
    # API Configuration
    REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT', '120'))
    API_MAX_RETRIES: int = int(os.getenv('API_MAX_RETRIES', '3'))
    API_RETRY_DELAY: int = int(os.getenv('API_RETRY_DELAY', '5'))
    API_BACKOFF_MULTIPLIER: float = float(os.getenv('API_BACKOFF_MULTIPLIER', '2'))
    
    # Performance Settings
    ENABLE_CACHING: bool = os.getenv('ENABLE_CACHING', 'True').lower() == 'true'
    CACHE_TIMEOUT: int = int(os.getenv('CACHE_TIMEOUT', '3600'))
    CACHE_MAX_LENGTH: int = int(os.getenv('CACHE_MAX_LENGTH', '50000'))
    
    # GPU Management
    ENABLE_GPU_MONITORING: bool = os.getenv('ENABLE_GPU_MONITORING', 'True').lower() == 'true'
    GPU_CLEANUP_THRESHOLD: int = int(os.getenv('GPU_CLEANUP_THRESHOLD', '85'))
    
    # Security Settings
    ENABLE_CORS: bool = os.getenv('ENABLE_CORS', 'False').lower() == 'true'
    CORS_ORIGINS: str = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000')
    ENABLE_RATE_LIMITING: bool = os.getenv('ENABLE_RATE_LIMITING', 'False').lower() == 'true'
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
    
    # Feature Flags
    ENABLE_EXPERIMENTAL_FEATURES: bool = os.getenv('ENABLE_EXPERIMENTAL_FEATURES', 'False').lower() == 'true'
    ENABLE_BATCH_PROCESSING: bool = os.getenv('ENABLE_BATCH_PROCESSING', 'True').lower() == 'true'
    ENABLE_EXPORT: bool = os.getenv('ENABLE_EXPORT', 'True').lower() == 'true'
    ENABLE_METRICS: bool = os.getenv('ENABLE_METRICS', 'True').lower() == 'true'
    
    # Development Settings  
    ENABLE_DEBUG_TOOLBAR: bool = os.getenv('ENABLE_DEBUG_TOOLBAR', 'False').lower() == 'true'
    ENABLE_PROFILING: bool = os.getenv('ENABLE_PROFILING', 'False').lower() == 'true'
    MOCK_API_RESPONSES: bool = os.getenv('MOCK_API_RESPONSES', 'False').lower() == 'true'
    
    @classmethod
    def validate_required_keys(cls) -> None:
        """
        Validate that required API keys are present.
        
        Raises:
            ValueError: If required API keys are missing
        """
        missing_keys = []
        
        if not cls.GOOGLE_API_KEY:
            missing_keys.append('GOOGLE_API_KEY')
            
        if not cls.TAVILY_API_KEY:
            missing_keys.append('TAVILY_API_KEY')
            
        if missing_keys:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_keys)}. "
                "Please check your .env file."
            )
    
    @classmethod
    def get_cors_origins(cls) -> list:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in cls.CORS_ORIGINS.split(',')]

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging() -> logging.Logger:
    """
    Configure application logging with file rotation and appropriate levels.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(__name__.split('.')[0])  # Root module name
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler (always present)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, Config.LOG_LEVEL))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if Config.LOG_FILE:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            Config.LOG_FILE,
            maxBytes=Config.LOG_MAX_SIZE,
            backupCount=Config.LOG_BACKUP_COUNT
        )
        file_handler.setLevel(getattr(logging, Config.LOG_LEVEL))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# =============================================================================
# INITIALIZATION
# =============================================================================

# Initialize logger
logger = setup_logging()

# Validate configuration on import (but allow override for testing)
if not os.getenv('SKIP_CONFIG_VALIDATION'):
    try:
        Config.validate_required_keys()
        logger.info("✅ Configuration validation passed")
    except ValueError as e:
        logger.warning(f"⚠️ Configuration validation failed: {e}")
        logger.info("💡 The application may still work with limited functionality")

# Export commonly used items
__all__ = [
    'Config',
    'logger',
    'DATABASE_PATH',
    'DEFAULT_EVALUATION_PROMPT',
    'setup_logging'
]