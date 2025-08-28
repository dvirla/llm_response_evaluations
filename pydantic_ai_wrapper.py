#!/usr/bin/env python3
"""
Pydantic AI Wrapper with GPU Memory Management

A comprehensive wrapper for pydantic_ai that provides:
- Integration with Ollama models via OpenAI-compatible API
- Tavily search tool for AI-optimized web search
- GPU memory management and model unloading capabilities
- Robust error handling and retry mechanisms
- Structured response formatting

This module enables seamless switching between different Ollama models
while providing intelligent memory management for efficient GPU utilization.

Author: dvirla
License: MIT
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)


class TavilySearchTool:
    """
    Tavily Search API integration for AI-optimized web search.
    
    Provides intelligent web search capabilities specifically designed for AI agents.
    Tavily's search API returns clean, structured results optimized for LLM consumption
    rather than traditional web search ranking algorithms.
    
    Attributes:
        api_key: Tavily API key from environment variables
        client: Initialized Tavily client instance
    """
    
    def __init__(self):
        self.api_key = os.getenv('TAVILY_API_KEY')
        self.client = None
        
        if self.api_key:
            try:
                from tavily import TavilyClient
                self.client = TavilyClient(api_key=self.api_key)
                logger.info("✅ Tavily Search API initialized successfully")
            except ImportError:
                logger.error("❌ tavily-python package not installed. Run: pip install tavily-python")
                self.client = None
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Tavily Search API: {e}")
                self.client = None
        else:
            logger.warning("⚠️ TAVILY_API_KEY not found in environment variables")

    def search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Execute AI-optimized search using Tavily API.
        
        Args:
            query: Search query string
            num_results: Maximum number of search results to return (default: 5)
            
        Returns:
            Dict containing search results in grounding format:
            - has_grounding: Boolean indicating search success
            - search_queries: List of queries executed
            - sources: List of result sources with titles and URLs
            - error: Error message if search fails
            
        Note:
            Uses "basic" search depth for faster results. Results are optimized
            for AI consumption with clean content extraction.
        """
        if not self.client:
            return {
                "has_grounding": False,
                "search_queries": [query],
                "sources": [],
                "error": "Tavily Search API not configured"
            }
        
        try:
            # Tavily search with AI-optimized parameters
            result = self.client.search(
                query=query,
                search_depth="basic",  # "basic" or "advanced" 
                max_results=num_results,
                include_answer=False,  # We want raw sources, not AI summary
                include_raw_content=False  # Clean content is better for LLMs
            )
            
            sources = []
            if 'results' in result and result['results']:
                for item in result['results'][:num_results]:
                    sources.append({
                        "title": item.get('title', 'Unknown'),
                        "uri": item.get('url', ''),
                        "snippet": item.get('content', '')
                    })
            
            grounding_info = {
                "has_grounding": len(sources) > 0,
                "search_queries": [query],
                "sources": sources
            }
            
            logger.info(f"🔍 Tavily Search found {len(sources)} results for: {query[:50]}...")
            return grounding_info
            
        except Exception as e:
            logger.error(f"❌ Tavily Search failed: {e}")
            return {
                "has_grounding": False,
                "search_queries": [query],
                "sources": [],
                "error": str(e)
            }


class PydanticAIOllamaWrapper:
    """
    Comprehensive wrapper for pydantic_ai with Ollama integration and GPU management.
    
    Provides seamless integration between pydantic_ai agents and Ollama models,
    with advanced features including:
    - Dynamic agent creation and caching
    - Tavily search integration for web-aware responses
    - GPU memory management and model unloading
    - Robust error handling with exponential backoff
    - Structured response formatting
    
    Attributes:
        base_url: Ollama API base URL (defaults to localhost:11434/v1)
        tavily_search: Tavily search tool instance
        _agents: Cache of created agents to avoid recreation overhead
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
        self.tavily_search = TavilySearchTool()
        
        # We'll create agents dynamically for each model/search combination
        self._agents = {}
        
    def _get_agent_without_search(self, model_name: str):
        """Get or create agent without search tools"""
        cache_key = f"{model_name}_no_search"
        if cache_key not in self._agents:
            try:
                # Create agent using model string format that pydantic_ai understands
                # The format should be 'ollama:model_name' but let me try the provider approach
                from pydantic_ai.models.openai import OpenAIChatModel
                from pydantic_ai.providers.ollama import OllamaProvider
                
                provider = OllamaProvider(base_url=self.base_url)
                model = OpenAIChatModel(model_name=model_name, provider=provider)
                
                agent = Agent(
                    model=model,
                    system_prompt=(
                        "You are a helpful AI assistant. Provide clear, accurate, and comprehensive responses. "
                        "Think step by step and provide your reasoning before giving your final answer."
                    )
                )
                
                self._agents[cache_key] = agent
                logger.info(f"✅ Created agent for {model_name} without search")
            except Exception as e:
                logger.error(f"❌ Failed to create agent for {model_name}: {e}")
                raise
                
        return self._agents[cache_key]
    
    def _get_agent_with_search(self, model_name: str):
        """Get or create agent with search tools"""
        cache_key = f"{model_name}_with_search"
        if cache_key not in self._agents:
            try:
                # Create agent using model string format that pydantic_ai understands
                from pydantic_ai.models.openai import OpenAIChatModel
                from pydantic_ai.providers.ollama import OllamaProvider
                
                provider = OllamaProvider(base_url=self.base_url)
                model = OpenAIChatModel(model_name=model_name, provider=provider)
                
                agent = Agent(
                    model=model,
                    system_prompt=(
                        "You are a helpful AI assistant with access to web search. "
                        "When answering questions, first consider if you need current information. "
                        "If so, use the search tool to get up-to-date information before providing your response. "
                        "Think step by step and provide your reasoning before giving your final answer."
                    )
                )
                
                # Add Tavily search tool
                @agent.tool
                async def web_search(ctx: RunContext, query: str) -> str:
                    """Search the web for current information using Tavily AI-optimized search"""
                    search_results = self.tavily_search.search(query)
                    
                    if not search_results.get("has_grounding", False):
                        return f"Search failed: {search_results.get('error', 'No results found')}"
                    
                    # Format search results for the model
                    formatted_results = []
                    for source in search_results["sources"][:3]:  # Limit to top 3 results
                        formatted_results.append(
                            f"Title: {source['title']}\n"
                            f"URL: {source['uri']}\n"
                            f"Content: {source['snippet']}\n"
                        )
                    
                    return "Search results:\n\n" + "\n---\n".join(formatted_results)
                
                self._agents[cache_key] = agent
                logger.info(f"✅ Created agent for {model_name} with search")
            except Exception as e:
                logger.error(f"❌ Failed to create agent with search for {model_name}: {e}")
                raise
                
        return self._agents[cache_key]
    
    async def query_ollama_without_search(self, prompt: str, model_name: str, max_retries: int = 3) -> Dict[str, Any]:
        """Query Ollama model without search tools - matches Gemini interface"""
        logger.info(f"🤔 Querying {model_name} WITHOUT search tools...")
        
        for attempt in range(max_retries):
            try:
                agent = self._get_agent_without_search(model_name)
                
                # Run the agent
                result = await agent.run(prompt)
                
                # Extract thinking and response - pydantic_ai doesn't separate these by default
                # so we'll simulate the structure to match Gemini's format
                if hasattr(result, 'data'):
                    full_response = str(result.data)
                elif hasattr(result, 'output'):
                    full_response = str(result.output)
                else:
                    full_response = str(result)
                
                # Simple heuristic to separate thinking from final response
                # Look for patterns like "Let me think..." or "First, I need to..."
                thinking = ""
                final_response = full_response
                
                # If response contains reasoning patterns, try to separate
                thinking_indicators = [
                    "Let me think", "First,", "To answer this", "I need to consider",
                    "Let me analyze", "Step by step", "My reasoning", "To solve this"
                ]
                
                lines = full_response.split('\n')
                thinking_lines = []
                response_lines = []
                in_thinking = False
                
                for line in lines:
                    if any(indicator.lower() in line.lower() for indicator in thinking_indicators):
                        in_thinking = True
                        thinking_lines.append(line)
                    elif in_thinking and (line.strip() == "" or line.startswith("Therefore") or line.startswith("In conclusion")):
                        thinking_lines.append(line)
                        if line.startswith(("Therefore", "In conclusion")):
                            in_thinking = False
                    elif in_thinking:
                        thinking_lines.append(line)
                    else:
                        response_lines.append(line)
                
                if thinking_lines:
                    thinking = '\n'.join(thinking_lines).strip()
                    final_response = '\n'.join(response_lines).strip()
                
                if not final_response.strip():
                    final_response = full_response
                
                logger.info(f"✅ {model_name} without search completed successfully")
                return {
                    "thinking": thinking,
                    "response": final_response,
                    "full_response": full_response
                }
                
            except Exception as e:
                attempt_msg = f"attempt {attempt + 1}/{max_retries}"
                logger.warning(f"⚠️ {model_name} without search failed ({attempt_msg}): {str(e)}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    delay = 5 * (2 ** attempt)
                    logger.info(f"⏳ Waiting {delay}s before retry...")
                    import asyncio
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ {model_name} without search failed after {max_retries} attempts: {str(e)}")
                    return {"error": f"Failed after {max_retries} attempts: {str(e)}"}
    
    async def query_ollama_with_search(self, prompt: str, model_name: str, max_retries: int = 3) -> Dict[str, Any]:
        """Query Ollama model with search tools - matches Gemini interface"""
        logger.info(f"🌐 Querying {model_name} WITH search tools...")
        
        search_queries_used = []
        grounding_info = None
        
        for attempt in range(max_retries):
            try:
                agent = self._get_agent_with_search(model_name)
                
                # Run the agent - it will automatically use search if needed
                result = await agent.run(prompt)
                
                # Extract the response
                if hasattr(result, 'data'):
                    full_response = str(result.data)
                elif hasattr(result, 'output'):
                    full_response = str(result.output)
                else:
                    full_response = str(result)
                
                # Try to extract search queries from the run messages
                if hasattr(result, 'all_messages'):
                    for message in result.all_messages():
                        if hasattr(message, 'parts'):
                            for part in message.parts:
                                if hasattr(part, 'tool_name') and part.tool_name == 'web_search':
                                    if hasattr(part, 'args'):
                                        # Safely handle args - it might be a string or dict
                                        try:
                                            if isinstance(part.args, dict) and 'query' in part.args:
                                                search_queries_used.append(part.args['query'])
                                            elif isinstance(part.args, str):
                                                # Sometimes args might be a JSON string
                                                import json
                                                args_dict = json.loads(part.args)
                                                if 'query' in args_dict:
                                                    search_queries_used.append(args_dict['query'])
                                        except (json.JSONDecodeError, TypeError, KeyError):
                                            # Skip if we can't parse args
                                            continue
                
                # If we used search, create grounding info
                if search_queries_used:
                    # Get search results for grounding info
                    search_results = self.tavily_search.search(search_queries_used[0] if search_queries_used else prompt)
                    # Ensure search_results is a dictionary before assigning
                    if isinstance(search_results, dict):
                        grounding_info = search_results.copy()  # Make a copy to avoid modifying original
                        grounding_info["search_queries"] = search_queries_used
                    else:
                        # If search_results is not a dict, create a basic structure
                        grounding_info = {
                            "has_grounding": False,
                            "search_queries": search_queries_used,
                            "sources": [],
                            "error": f"Invalid search results format: {type(search_results)}"
                        }
                
                # Parse thinking vs response (similar to without_search)
                thinking = ""
                final_response = full_response
                
                thinking_indicators = [
                    "Let me search", "I should search", "I need to find", "Let me look up",
                    "Let me think", "First,", "To answer this", "I need to consider"
                ]
                
                lines = full_response.split('\n')
                thinking_lines = []
                response_lines = []
                in_thinking = False
                
                for line in lines:
                    if any(indicator.lower() in line.lower() for indicator in thinking_indicators):
                        in_thinking = True
                        thinking_lines.append(line)
                    elif in_thinking and (line.strip() == "" or line.startswith("Based on") or line.startswith("According to")):
                        thinking_lines.append(line)
                        if line.startswith(("Based on", "According to")):
                            in_thinking = False
                    elif in_thinking:
                        thinking_lines.append(line)
                    else:
                        response_lines.append(line)
                
                if thinking_lines:
                    thinking = '\n'.join(thinking_lines).strip()
                    final_response = '\n'.join(response_lines).strip()
                
                if not final_response.strip():
                    final_response = full_response
                
                logger.info(f"✅ {model_name} with search completed successfully")
                return {
                    "thinking": thinking,
                    "response": final_response,
                    "full_response": full_response,
                    "grounding_info": grounding_info
                }
                
            except Exception as e:
                attempt_msg = f"attempt {attempt + 1}/{max_retries}"
                logger.warning(f"⚠️ {model_name} with search failed ({attempt_msg}): {str(e)}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    delay = 5 * (2 ** attempt)
                    logger.info(f"⏳ Waiting {delay}s before retry...")
                    import asyncio
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ {model_name} with search failed after {max_retries} attempts: {str(e)}")
                    return {"error": f"Failed after {max_retries} attempts: {str(e)}"}

    def query_ollama_without_search_sync(self, prompt: str, model_name: str, max_retries: int = 3) -> Dict[str, Any]:
        """Synchronous wrapper for query_ollama_without_search"""
        import asyncio
        return asyncio.run(self.query_ollama_without_search(prompt, model_name, max_retries))
    
    def query_ollama_with_search_sync(self, prompt: str, model_name: str, max_retries: int = 3) -> Dict[str, Any]:
        """Synchronous wrapper for query_ollama_with_search"""
        import asyncio
        return asyncio.run(self.query_ollama_with_search(prompt, model_name, max_retries))
    
    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """
        Unload a specific model from GPU memory using Ollama's keep_alive parameter.
        
        Sends a request to Ollama's /api/chat endpoint with keep_alive=0 to
        immediately unload the specified model from GPU memory, freeing up VRAM
        for other models or applications.
        
        Args:
            model_name: Name of the Ollama model to unload (e.g., 'llama3.1:70b')
            
        Returns:
            Dict containing:
            - success: Boolean indicating if unload was successful
            - message: Success message or error description
            - error: Error message if unload failed
            
        Note:
            This uses Ollama's native memory management API rather than
            external GPU control tools, ensuring compatibility and safety.
        """
        logger.info(f"🧹 Attempting to unload {model_name} from GPU memory...")
        
        try:
            import httpx
            
            # Use Ollama's /api/chat endpoint with keep_alive=0 to unload the model
            unload_url = self.base_url.replace("/v1", "") + "/api/chat"
            
            response = httpx.post(
                unload_url,
                json={
                    "model": model_name,
                    "keep_alive": 0,  # This tells Ollama to unload the model immediately
                    "messages": []  # Empty messages array - we're just unloading
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Successfully unloaded {model_name} from GPU memory")
                return {"success": True, "message": f"Model {model_name} unloaded from GPU memory"}
            else:
                logger.warning(f"⚠️ Unload request returned status {response.status_code}: {response.text}")
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error(f"❌ Failed to unload {model_name}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def unload_all_models(self) -> Dict[str, Any]:
        """
        Unload all currently loaded models from GPU memory with batch processing.
        
        Uses 'ollama ps' to discover loaded models, then systematically unloads
        each one using the keep_alive=0 parameter. Provides detailed results
        for each model's unload operation.
        
        Returns:
            Dict containing:
            - success: Boolean indicating if any models were successfully unloaded
            - message: Summary of results (e.g., "Unloaded 3/4 models")
            - results: Array of per-model results with model name, success status, and message
            
        Note:
            If no models are currently loaded, returns success=True with appropriate message.
            Individual model failures don't prevent attempting to unload other models.
        """
        logger.info("🧹 Attempting to unload all models from GPU memory...")
        
        try:
            import subprocess
            
            # First, get list of currently loaded models using ollama ps
            result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.warning(f"⚠️ Could not get loaded models: {result.stderr}")
                return {"success": False, "error": "Could not retrieve loaded models"}
            
            # Parse the output to extract loaded model names
            lines = result.stdout.strip().split('\n')
            loaded_models = []
            
            for line in lines[1:]:  # Skip header line
                if line.strip():
                    # Extract model name (first column)
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        loaded_models.append(model_name)
            
            if not loaded_models:
                logger.info("✅ No models currently loaded in memory")
                return {"success": True, "message": "No models currently loaded"}
            
            # Unload each model
            unload_results = []
            for model_name in loaded_models:
                result = self.unload_model(model_name)
                unload_results.append({
                    "model": model_name,
                    "success": result.get("success", False),
                    "message": result.get("message", result.get("error", "Unknown error"))
                })
            
            successful_unloads = sum(1 for r in unload_results if r["success"])
            total_models = len(unload_results)
            
            logger.info(f"🧹 Unloaded {successful_unloads}/{total_models} models from GPU memory")
            
            return {
                "success": successful_unloads > 0,
                "message": f"Unloaded {successful_unloads}/{total_models} models",
                "results": unload_results
            }
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout while checking loaded models")
            return {"success": False, "error": "Timeout while checking loaded models"}
        except FileNotFoundError:
            logger.error("❌ Ollama command not found")
            return {"success": False, "error": "Ollama command not found"}
        except Exception as e:
            logger.error(f"❌ Failed to unload all models: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """Get list of currently loaded models using ollama ps"""
        try:
            import subprocess
            
            result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.warning(f"⚠️ Could not get loaded models: {result.stderr}")
                return {"success": False, "error": "Could not retrieve loaded models"}
            
            # Parse the output
            lines = result.stdout.strip().split('\n')
            loaded_models = []
            
            for line in lines[1:]:  # Skip header line
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:  # NAME ID SIZE PROCESSOR [UNTIL]
                        model_info = {
                            "name": parts[0],
                            "id": parts[1],
                            "size": parts[2],
                            "processor": parts[3],
                            "until": " ".join(parts[4:]) if len(parts) > 4 else "Unknown"
                        }
                        loaded_models.append(model_info)
            
            logger.info(f"📊 Found {len(loaded_models)} loaded models")
            return {"success": True, "models": loaded_models}
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout while checking loaded models")
            return {"success": False, "error": "Timeout while checking loaded models"}
        except FileNotFoundError:
            logger.error("❌ Ollama command not found")
            return {"success": False, "error": "Ollama command not found"}
        except Exception as e:
            logger.error(f"❌ Failed to get loaded models: {str(e)}")
            return {"success": False, "error": str(e)}