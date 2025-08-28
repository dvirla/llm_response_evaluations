#!/usr/bin/env python3
"""
Gemini Service Module - AI Model Response Evaluator

Dedicated Google Gemini 2.5 Pro integration service.
Handles all Gemini-specific operations including dual-mode querying,
response parsing, grounding information processing, and error handling.

Author: dvirla
License: MIT
"""

import time
from typing import Dict, Any
from google import genai
from google.genai import types
from config import Config, logger

class GeminiService:
    """
    Dedicated Google Gemini 2.5 Pro integration service.
    
    Provides clean separation of Gemini-specific functionality including:
    - Direct Google GenAI API integration
    - Dual-mode querying (with/without Google Search)
    - Thinking process extraction and parsing
    - Grounding metadata processing
    - Comprehensive error handling with retry logic
    
    Attributes:
        client: Google GenAI API client instance
    """
    
    def __init__(self):
        """
        Initialize Gemini service with Google GenAI client.
        
        Raises:
            ValueError: If GOOGLE_API_KEY is not configured
            ConnectionError: If API client initialization fails
        """
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for Gemini service")
            
        try:
            self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
            logger.info("✅ Gemini service initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {str(e)}")
            raise ConnectionError(f"Gemini client initialization failed: {str(e)}")
    
    def query_without_search(self, prompt: str, max_retries: int = None) -> Dict[str, Any]:
        """
        Query Gemini 2.5 Pro without search tools and return structured response.
        
        Implements exponential backoff retry logic and comprehensive error handling
        for robust operation under various network and API conditions.
        
        Args:
            prompt: User query/prompt to send to Gemini
            max_retries: Maximum number of retry attempts (uses config default if None)
            
        Returns:
            Dict containing 'thinking', 'response', 'full_response' keys on success,
            or 'error' key on failure
            
        Note:
            Uses ThinkingConfig to capture Gemini's reasoning process alongside
            the final response for transparency and debugging.
        """
        if max_retries is None:
            max_retries = Config.API_MAX_RETRIES
            
        logger.info("🤔 Querying Gemini WITHOUT search tools...")
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(include_thoughts=True)
                    )
                )
                
                # Validate response structure
                if not self._validate_response(response):
                    raise ValueError("Invalid or empty response from Gemini API")
                
                # Parse thinking and response content
                thinking, final_response = self._parse_response_content(response)
                
                # Ensure we got meaningful content
                if not thinking.strip() and not final_response.strip():
                    raise ValueError("Gemini returned empty response content")
                
                logger.info("✅ Gemini without search completed successfully")
                return {
                    "thinking": thinking.strip(),
                    "response": final_response.strip(),
                    "full_response": response.text if hasattr(response, 'text') else final_response.strip()
                }
                
            except Exception as e:
                attempt_msg = f"attempt {attempt + 1}/{max_retries}"
                logger.warning(f"⚠️ Gemini without search failed ({attempt_msg}): {str(e)}")
                
                if attempt < max_retries - 1:
                    delay = self._calculate_retry_delay(attempt)
                    logger.info(f"⏳ Waiting {delay}s before retry...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Gemini without search failed after {max_retries} attempts: {str(e)}")
                    return {"error": f"Failed after {max_retries} attempts: {str(e)}"}
    
    def query_with_search(self, prompt: str, max_retries: int = None) -> Dict[str, Any]:
        """
        Query Gemini 2.5 Pro with Google Search tool and return structured response.
        
        Enables Google Search integration for up-to-date information and web-grounded
        responses. Includes grounding metadata for transparency and verification.
        
        Args:
            prompt: User query/prompt to send to Gemini
            max_retries: Maximum number of retry attempts (uses config default if None)
            
        Returns:
            Dict containing 'thinking', 'response', 'full_response', 'grounding_info' keys
            on success, or 'error' key on failure
            
        Note:
            Grounding information includes search queries executed and source metadata
            for transparency and fact-checking purposes.
        """
        if max_retries is None:
            max_retries = Config.API_MAX_RETRIES
            
        logger.info("🌐 Querying Gemini WITH search tools...")
        
        for attempt in range(max_retries):
            try:
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                config = types.GenerateContentConfig(
                    tools=[grounding_tool],
                    thinking_config=types.ThinkingConfig(include_thoughts=True)
                )
                
                response = self.client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt,
                    config=config,
                )
                
                # Validate response structure
                if not self._validate_response(response):
                    raise ValueError("Invalid or empty response from Gemini API")
                
                # Parse thinking and response content
                thinking, final_response = self._parse_response_content(response)
                
                # Ensure we got meaningful content
                if not thinking.strip() and not final_response.strip():
                    raise ValueError("Gemini returned empty response content")
                
                # Extract grounding information
                grounding_info = self._extract_grounding_info(response)
                
                logger.info("✅ Gemini with search completed successfully")
                return {
                    "thinking": thinking.strip(),
                    "response": final_response.strip(),
                    "full_response": response.text if hasattr(response, 'text') else final_response.strip(),
                    "grounding_info": grounding_info
                }
                
            except Exception as e:
                attempt_msg = f"attempt {attempt + 1}/{max_retries}"
                logger.warning(f"⚠️ Gemini with search failed ({attempt_msg}): {str(e)}")
                
                if attempt < max_retries - 1:
                    delay = self._calculate_retry_delay(attempt)
                    logger.info(f"⏳ Waiting {delay}s before retry...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Gemini with search failed after {max_retries} attempts: {str(e)}")
                    return {"error": f"Failed after {max_retries} attempts: {str(e)}"}
    
    def _validate_response(self, response) -> bool:
        """
        Validate that Gemini response has expected structure.
        
        Args:
            response: Raw response from Gemini API
            
        Returns:
            bool: True if response is valid, False otherwise
        """
        return (response and 
                hasattr(response, 'candidates') and 
                response.candidates and
                response.candidates[0] and 
                hasattr(response.candidates[0], 'content') and
                response.candidates[0].content and 
                hasattr(response.candidates[0].content, 'parts') and
                response.candidates[0].content.parts)
    
    def _parse_response_content(self, response) -> tuple[str, str]:
        """
        Parse Gemini response to extract thinking and final response content.
        
        Args:
            response: Validated Gemini API response
            
        Returns:
            tuple: (thinking_content, final_response_content)
        """
        thinking = ""
        final_response = ""
        
        for part in response.candidates[0].content.parts:
            if not part or not hasattr(part, 'text') or not part.text:
                continue
                
            if hasattr(part, 'thought') and part.thought:
                thinking += part.text + "\n"
            else:
                final_response += part.text + "\n"
        
        return thinking, final_response
    
    def _extract_grounding_info(self, response) -> Dict[str, Any]:
        """
        Extract grounding metadata from Gemini search-enabled response.
        
        Args:
            response: Gemini API response with potential grounding metadata
            
        Returns:
            Dict: Structured grounding information or None if not available
        """
        try:
            if not (hasattr(response.candidates[0], 'grounding_metadata') and 
                   response.candidates[0].grounding_metadata):
                return None
            
            grounding_metadata = response.candidates[0].grounding_metadata
            grounding_info = {
                "has_grounding": True,
                "search_queries": [],
                "sources": []
            }
            
            # Extract search queries
            if hasattr(grounding_metadata, 'web_search_queries'):
                grounding_info["search_queries"] = grounding_metadata.web_search_queries
            
            # Extract source information
            if hasattr(grounding_metadata, 'grounding_chunks'):
                grounding_info["sources"] = [
                    {
                        "title": chunk.web.title if (hasattr(chunk, 'web') and 
                                                   hasattr(chunk.web, 'title')) else "Unknown",
                        "uri": chunk.web.uri if (hasattr(chunk, 'web') and 
                                               hasattr(chunk.web, 'uri')) else ""
                    }
                    for chunk in grounding_metadata.grounding_chunks[:5]  # Limit to first 5 sources
                ]
            
            return grounding_info
            
        except Exception as grounding_error:
            logger.warning(f"⚠️ Failed to parse grounding metadata: {str(grounding_error)}")
            return None
    
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Gemini model and service status.
        
        Returns:
            Dict: Model information and service status
        """
        return {
            "model_name": "gemini-2.5-pro",
            "provider": "Google",
            "supports_search": True,
            "supports_thinking": True,
            "max_retries": Config.API_MAX_RETRIES,
            "retry_delay": Config.API_RETRY_DELAY,
            "service_status": "active"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the Gemini service.
        
        Returns:
            Dict: Health status and basic connectivity test results
        """
        try:
            # Simple test query
            test_response = self.query_without_search("Hello", max_retries=1)
            
            return {
                "status": "healthy" if "error" not in test_response else "unhealthy",
                "api_accessible": "error" not in test_response,
                "last_check": time.time(),
                "error": test_response.get("error")
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "api_accessible": False,
                "last_check": time.time(),
                "error": str(e)
            }

# Export the service class
__all__ = ['GeminiService']