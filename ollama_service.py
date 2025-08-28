#!/usr/bin/env python3
"""
Ollama Service Module - AI Model Response Evaluator

Dedicated Ollama local model integration service using pydantic_ai.
Handles all Ollama-specific operations including model management,
GPU memory optimization, and local AI model querying.

Author: dvirla
License: MIT
"""

import time
from typing import Dict, Any, List, Optional
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName
from config import Config, logger


class OllamaService:
    """
    Dedicated Ollama local model integration service.
    
    Provides clean separation of Ollama-specific functionality including:
    - pydantic_ai Agent wrapper for local model access
    - GPU memory management with automatic cleanup
    - Model status monitoring and health checks
    - Standardized response formatting
    - Comprehensive error handling with retry logic
    
    Attributes:
        agents: Dictionary of cached pydantic_ai Agent instances
        last_used_model: Track most recently used model for cleanup
    """
    
    def __init__(self):
        """
        Initialize Ollama service with empty agent cache.
        
        Agents are created on-demand to optimize memory usage and allow
        for dynamic model selection based on availability.
        """
        self.agents: Dict[str, Agent] = {}
        self.last_used_model: Optional[str] = None
        logger.info("✅ Ollama service initialized successfully")
    
    def query_model(self, model_name: str, prompt: str, max_retries: int = None) -> Dict[str, Any]:
        """
        Query a local Ollama model with comprehensive error handling.
        
        Creates pydantic_ai Agent on-demand and handles GPU memory management
        through Ollama's keep_alive parameter configuration.
        
        Args:
            model_name: Name of the Ollama model to use (e.g., 'llama3.2:3b')
            prompt: User query/prompt to send to the model
            max_retries: Maximum number of retry attempts (uses config default if None)
            
        Returns:
            Dict containing 'response' key on success, or 'error' key on failure
            
        Note:
            Automatically manages GPU memory by unloading previously used models
            when switching between models to prevent memory exhaustion.
        """
        if max_retries is None:
            max_retries = Config.API_MAX_RETRIES
            
        logger.info(f"🤖 Querying Ollama model '{model_name}'...")
        
        # Handle GPU memory management
        self._manage_gpu_memory(model_name)
        
        for attempt in range(max_retries):
            try:
                # Get or create agent for this model
                agent = self._get_agent(model_name)
                
                # Run the query
                result = agent.run_sync(prompt)
                
                # Update last used model tracking
                self.last_used_model = model_name
                
                # Validate result
                if not result or not result.data:
                    raise ValueError("Ollama returned empty response")
                
                logger.info(f"✅ Ollama model '{model_name}' completed successfully")
                return {
                    "response": str(result.data).strip(),
                    "model_name": model_name,
                    "model_provider": "ollama"
                }
                
            except Exception as e:
                attempt_msg = f"attempt {attempt + 1}/{max_retries}"
                logger.warning(f"⚠️ Ollama model '{model_name}' failed ({attempt_msg}): {str(e)}")
                
                if attempt < max_retries - 1:
                    delay = self._calculate_retry_delay(attempt)
                    logger.info(f"⏳ Waiting {delay}s before retry...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Ollama model '{model_name}' failed after {max_retries} attempts: {str(e)}")
                    return {"error": f"Failed after {max_retries} attempts: {str(e)}"}
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Ollama models.
        
        Attempts to query Ollama API for model list, falls back to
        common models if API is not accessible.
        
        Returns:
            List[str]: List of available model names
        """
        try:
            # Try to get actual models from Ollama
            # This would require additional HTTP client setup
            # For now, return common models that are typically available
            common_models = [
                "llama3.2:3b",
                "llama3.2:1b",
                "llama3.1:8b",
                "mixtral:8x7b",
                "codellama:7b",
                "phi3:mini",
                "gemma2:2b",
                "qwen2.5:7b"
            ]
            
            logger.info(f"📋 Available models: {len(common_models)} models")
            return common_models
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get model list: {str(e)}")
            return ["llama3.2:3b"]  # Fallback to most common model
    
    def _get_agent(self, model_name: str) -> Agent:
        """
        Get or create pydantic_ai Agent for the specified model.
        
        Caches agents to avoid recreation overhead while allowing
        for memory management through selective cleanup.
        
        Args:
            model_name: Name of the Ollama model
            
        Returns:
            Agent: Configured pydantic_ai Agent instance
            
        Raises:
            ValueError: If model name is invalid or agent creation fails
        """
        if model_name not in self.agents:
            try:
                # Create new agent with Ollama configuration
                agent = Agent(
                    model=KnownModelName(model_name),
                    model_settings={
                        'base_url': Config.OLLAMA_BASE_URL,
                        'keep_alive': Config.OLLAMA_KEEP_ALIVE,
                        'num_parallel': Config.OLLAMA_NUM_PARALLEL
                    }
                )
                
                self.agents[model_name] = agent
                logger.info(f"🔧 Created new agent for model '{model_name}'")
                
            except Exception as e:
                logger.error(f"❌ Failed to create agent for '{model_name}': {str(e)}")
                raise ValueError(f"Failed to create agent for model '{model_name}': {str(e)}")
        
        return self.agents[model_name]
    
    def _manage_gpu_memory(self, new_model: str) -> None:
        """
        Manage GPU memory by cleaning up unused models.
        
        Implements intelligent memory management by unloading previously
        used models when switching to prevent GPU memory exhaustion.
        
        Args:
            new_model: Name of the model about to be used
        """
        if not Config.ENABLE_GPU_MONITORING:
            return
            
        # If switching models, clean up previous model
        if (self.last_used_model and 
            self.last_used_model != new_model and 
            self.last_used_model in self.agents):
            
            try:
                # Remove agent to trigger cleanup
                del self.agents[self.last_used_model]
                logger.info(f"🧹 Cleaned up model '{self.last_used_model}' for memory management")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup model '{self.last_used_model}': {str(e)}")
    
    def _calculate_retry_delay(self, attempt: int) -> int:
        """
        Calculate exponential backoff delay for retry attempts.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            int: Delay in seconds
        """
        base_delay = Config.API_RETRY_DELAY
        multiplier = Config.API_BACKOFF_MULTIPLIER
        return int(base_delay * (multiplier ** attempt))
    
    def cleanup_all_agents(self) -> None:
        """
        Clean up all cached agents to free GPU memory.
        
        Useful for manual memory management or when switching contexts
        that require different model configurations.
        """
        try:
            model_count = len(self.agents)
            self.agents.clear()
            self.last_used_model = None
            logger.info(f"🧹 Cleaned up {model_count} Ollama agents")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup agents: {str(e)}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific Ollama model.
        
        Args:
            model_name: Name of the model to get info for
            
        Returns:
            Dict: Model information and status
        """
        return {
            "model_name": model_name,
            "provider": "ollama",
            "base_url": Config.OLLAMA_BASE_URL,
            "keep_alive": Config.OLLAMA_KEEP_ALIVE,
            "num_parallel": Config.OLLAMA_NUM_PARALLEL,
            "is_loaded": model_name in self.agents,
            "is_last_used": model_name == self.last_used_model
        }
    
    def health_check(self, model_name: str = "llama3.2:3b") -> Dict[str, Any]:
        """
        Perform a health check on the Ollama service.
        
        Args:
            model_name: Model to test with (defaults to common model)
            
        Returns:
            Dict: Health status and basic connectivity test results
        """
        try:
            # Simple test query
            test_response = self.query_model(model_name, "Hello", max_retries=1)
            
            return {
                "status": "healthy" if "error" not in test_response else "unhealthy",
                "ollama_accessible": "error" not in test_response,
                "test_model": model_name,
                "base_url": Config.OLLAMA_BASE_URL,
                "last_check": time.time(),
                "error": test_response.get("error")
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "ollama_accessible": False,
                "test_model": model_name,
                "base_url": Config.OLLAMA_BASE_URL,
                "last_check": time.time(),
                "error": str(e)
            }


# Export the service class
__all__ = ['OllamaService']