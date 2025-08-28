#!/usr/bin/env python3
"""
Evaluation Service Module - AI Model Response Evaluator

Dedicated evaluation service using Ollama models for AI response assessment.
Handles all evaluation-specific operations including custom prompt processing,
response comparison, and evaluation result formatting.

Author: dvirla
License: MIT
"""

import time
from typing import Dict, Any, Optional
from ollama_service import OllamaService
from config import Config, logger, DEFAULT_EVALUATION_PROMPT


class EvaluationService:
    """
    Dedicated evaluation service for AI response assessment.
    
    Provides clean separation of evaluation-specific functionality including:
    - Custom evaluation prompt template processing
    - Ollama model integration for evaluation tasks
    - Structured evaluation result formatting
    - Response comparison and analysis
    - Comprehensive error handling and retry logic
    
    Attributes:
        ollama_service: OllamaService instance for model querying
        default_prompt_template: Default evaluation prompt template
    """
    
    def __init__(self, ollama_service: Optional[OllamaService] = None):
        """
        Initialize evaluation service with Ollama integration.
        
        Args:
            ollama_service: Optional pre-configured OllamaService instance.
                           If None, creates a new instance.
        """
        self.ollama_service = ollama_service or OllamaService()
        self.default_prompt_template = DEFAULT_EVALUATION_PROMPT
        logger.info("✅ Evaluation service initialized successfully")
    
    def evaluate_responses(self, original_prompt: str, gemini_without_search: Dict[str, Any], 
                          gemini_with_search: Dict[str, Any], evaluation_model: str,
                          custom_evaluation_prompt: Optional[str] = None,
                          max_retries: int = None) -> Dict[str, Any]:
        """
        Evaluate AI responses using a local Ollama model.
        
        Processes both Gemini response modes through a local evaluation model
        to provide comparative analysis and quality assessment.
        
        Args:
            original_prompt: The original user query that generated the responses
            gemini_without_search: Response data from Gemini without search
            gemini_with_search: Response data from Gemini with search
            evaluation_model: Name of the Ollama model to use for evaluation
            custom_evaluation_prompt: Optional custom evaluation prompt template
            max_retries: Maximum number of retry attempts (uses config default if None)
            
        Returns:
            Dict containing 'evaluation', 'used_without_search', 'used_with_search' keys
            on success, or 'error' key on failure
            
        Note:
            Automatically determines which response modes to include based on
            error status and content availability.
        """
        if max_retries is None:
            max_retries = Config.API_MAX_RETRIES
            
        logger.info(f"🔍 Starting evaluation with model '{evaluation_model}'...")
        
        # Determine which responses to include in evaluation
        use_without_search = self._should_use_response(gemini_without_search)
        use_with_search = self._should_use_response(gemini_with_search)
        
        if not use_without_search and not use_with_search:
            error_msg = "No valid responses available for evaluation"
            logger.error(f"❌ {error_msg}")
            return {"error": error_msg}
        
        # Build evaluation prompt
        evaluation_prompt = self._build_evaluation_prompt(
            original_prompt=original_prompt,
            gemini_without_search=gemini_without_search if use_without_search else None,
            gemini_with_search=gemini_with_search if use_with_search else None,
            custom_template=custom_evaluation_prompt
        )
        
        # Perform evaluation using Ollama service
        for attempt in range(max_retries):
            try:
                result = self.ollama_service.query_model(
                    model_name=evaluation_model,
                    prompt=evaluation_prompt,
                    max_retries=1  # Let this service handle retries
                )
                
                if "error" in result:
                    raise Exception(result["error"])
                
                logger.info(f"✅ Evaluation completed successfully with '{evaluation_model}'")
                return {
                    "evaluation": result["response"],
                    "used_without_search": use_without_search,
                    "used_with_search": use_with_search,
                    "evaluation_model": evaluation_model,
                    "evaluation_provider": "ollama"
                }
                
            except Exception as e:
                attempt_msg = f"attempt {attempt + 1}/{max_retries}"
                logger.warning(f"⚠️ Evaluation failed ({attempt_msg}): {str(e)}")
                
                if attempt < max_retries - 1:
                    delay = self._calculate_retry_delay(attempt)
                    logger.info(f"⏳ Waiting {delay}s before retry...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Evaluation failed after {max_retries} attempts: {str(e)}")
                    return {"error": f"Failed after {max_retries} attempts: {str(e)}"}
    
    def _should_use_response(self, response_data: Dict[str, Any]) -> bool:
        """
        Determine if a response should be included in evaluation.
        
        Args:
            response_data: Response data dictionary
            
        Returns:
            bool: True if response should be included, False otherwise
        """
        if not response_data:
            return False
            
        # Check for error conditions
        if response_data.get("error"):
            return False
            
        # Check for empty responses
        response_text = response_data.get("response", "").strip()
        if not response_text:
            return False
            
        return True
    
    def _build_evaluation_prompt(self, original_prompt: str, 
                                gemini_without_search: Optional[Dict[str, Any]],
                                gemini_with_search: Optional[Dict[str, Any]],
                                custom_template: Optional[str] = None) -> str:
        """
        Build the evaluation prompt from template and response data.
        
        Args:
            original_prompt: The original user query
            gemini_without_search: Optional response data without search
            gemini_with_search: Optional response data with search
            custom_template: Optional custom evaluation template
            
        Returns:
            str: Formatted evaluation prompt ready for model consumption
        """
        template = custom_template or self.default_prompt_template
        
        # Prepare response content with fallbacks
        without_search_thinking = ""
        without_search_response = "[Not available - response failed or was empty]"
        with_search_thinking = ""
        with_search_response = "[Not available - response failed or was empty]"
        
        if gemini_without_search:
            without_search_thinking = gemini_without_search.get("thinking", "")
            without_search_response = gemini_without_search.get("response", "")
        
        if gemini_with_search:
            with_search_thinking = gemini_with_search.get("thinking", "")
            with_search_response = gemini_with_search.get("response", "")
            
            # Add grounding information if available
            grounding_info = gemini_with_search.get("grounding_info")
            if grounding_info and grounding_info.get("has_grounding"):
                sources = grounding_info.get("sources", [])
                if sources:
                    source_info = "\\n\\nSources used:"
                    for i, source in enumerate(sources[:3], 1):  # Limit to 3 sources
                        source_info += f"\\n{i}. {source.get('title', 'Unknown')} - {source.get('uri', '')}"
                    with_search_response += source_info
        
        # Format the template
        try:
            formatted_prompt = template.format(
                original_prompt=original_prompt,
                gemini_without_search_thinking=without_search_thinking,
                gemini_without_search_response=without_search_response,
                gemini_with_search_thinking=with_search_thinking,
                gemini_with_search_response=with_search_response
            )
        except KeyError as e:
            logger.warning(f"⚠️ Template formatting error: {str(e)}, using simple format")
            # Fallback to simple format
            formatted_prompt = f"""Please evaluate these AI responses to the prompt: "{original_prompt}"

Response 1: {without_search_response}

Response 2: {with_search_response}

Please provide a comparative evaluation focusing on accuracy, completeness, and usefulness."""
        
        return formatted_prompt
    
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
    
    def get_available_models(self) -> list:
        """
        Get list of available evaluation models from Ollama service.
        
        Returns:
            List[str]: List of available model names for evaluation
        """
        return self.ollama_service.get_available_models()
    
    def validate_evaluation_model(self, model_name: str) -> Dict[str, Any]:
        """
        Validate that an evaluation model is available and functional.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Dict: Validation results with status and error information
        """
        try:
            available_models = self.get_available_models()
            
            if model_name not in available_models:
                return {
                    "valid": False,
                    "error": f"Model '{model_name}' not found in available models",
                    "available_models": available_models
                }
            
            # Test the model with a simple query
            health_check = self.ollama_service.health_check(model_name)
            
            return {
                "valid": health_check.get("status") == "healthy",
                "status": health_check.get("status"),
                "error": health_check.get("error"),
                "model_info": self.ollama_service.get_model_info(model_name)
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation failed: {str(e)}"
            }
    
    def get_default_evaluation_prompt(self) -> str:
        """
        Get the default evaluation prompt template.
        
        Returns:
            str: Default evaluation prompt template
        """
        return self.default_prompt_template
    
    def cleanup_resources(self) -> None:
        """
        Clean up evaluation service resources.
        
        Delegates to Ollama service for GPU memory cleanup.
        """
        try:
            self.ollama_service.cleanup_all_agents()
            logger.info("🧹 Evaluation service resources cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup evaluation resources: {str(e)}")


# Export the service class
__all__ = ['EvaluationService']