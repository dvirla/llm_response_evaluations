#!/usr/bin/env python3
"""
Routes Module - AI Model Response Evaluator

Flask Blueprint-based API routes for the AI Model Response Evaluator.
Handles all HTTP endpoints with proper separation of concerns and
comprehensive error handling.

Author: dvirla
License: MIT
"""

import json
import traceback
from flask import Blueprint, request, jsonify, render_template
from typing import Dict, Any, Optional

# Import services
from gemini_service import GeminiService
from ollama_service import OllamaService
from evaluation_service import EvaluationService
from models import (
    save_gemini_responses, save_evaluation, get_gemini_responses_history,
    get_gemini_response_by_id, get_evaluations_for_response, get_database_stats
)
from config import Config, logger

# Create Blueprint
api_bp = Blueprint('api', __name__)

# Initialize services (will be injected by main app)
gemini_service: Optional[GeminiService] = None
ollama_service: Optional[OllamaService] = None
evaluation_service: Optional[EvaluationService] = None


def init_services(gemini_svc: GeminiService, ollama_svc: OllamaService, eval_svc: EvaluationService):
    """
    Initialize services for the routes module.
    
    Args:
        gemini_svc: Configured GeminiService instance
        ollama_svc: Configured OllamaService instance  
        eval_svc: Configured EvaluationService instance
    """
    global gemini_service, ollama_service, evaluation_service
    gemini_service = gemini_svc
    ollama_service = ollama_svc
    evaluation_service = eval_svc
    logger.info("✅ Routes services initialized successfully")


# =============================================================================
# WEB INTERFACE ROUTES
# =============================================================================

@api_bp.route('/')
def index():
    """
    Main web interface for the AI Model Response Evaluator.
    
    Returns:
        Rendered HTML template with application interface
    """
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"❌ Failed to render index template: {str(e)}")
        return jsonify({"error": "Template rendering failed"}), 500


# =============================================================================
# AI RESPONSE GENERATION ROUTES
# =============================================================================

@api_bp.route('/api/query', methods=['POST'])
@api_bp.route('/api/query-gemini', methods=['POST'])
def query_gemini():
    """
    Generate responses using Gemini in both search modes.
    
    Request JSON:
        {
            "prompt": "string",
            "model_name": "string" (optional, defaults to gemini-2.5-pro)
        }
    
    Returns:
        JSON with response data or error information
    """
    try:
        if not gemini_service:
            return jsonify({"error": "Gemini service not initialized"}), 500
            
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400
        
        model_name = data.get('model_name', 'gemini-2.5-pro')
        
        logger.info(f"🔄 Processing query request (model: {model_name})")
        
        # Query both modes in parallel conceptually
        result_without_search = gemini_service.query_without_search(prompt)
        result_with_search = gemini_service.query_with_search(prompt)
        
        # Save to database
        response_id = save_gemini_responses(prompt, result_without_search, result_with_search, model_name)
        
        response_data = {
            "success": True,
            "response_id": response_id,
            "model_name": model_name,
            "prompt": prompt,
            "response_without_search": result_without_search,
            "response_with_search": result_with_search
        }
        
        logger.info(f"✅ Query processed successfully (ID: {response_id})")
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Query processing failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500


# =============================================================================
# EVALUATION ROUTES
# =============================================================================

@api_bp.route('/api/evaluate', methods=['POST'])
@api_bp.route('/api/evaluate-response', methods=['POST'])
def evaluate_response():
    """
    Evaluate stored Gemini responses using Ollama models.
    
    Request JSON:
        {
            "response_id": integer,
            "evaluation_model": "string",
            "custom_evaluation_prompt": "string" (optional)
        }
    
    Returns:
        JSON with evaluation results or error information
    """
    try:
        if not evaluation_service:
            return jsonify({"error": "Evaluation service not initialized"}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400
        
        response_id = data.get('response_id')
        evaluation_model = data.get('evaluation_model')
        custom_prompt = data.get('custom_evaluation_prompt')
        
        if not response_id or not evaluation_model:
            return jsonify({"error": "Missing 'response_id' or 'evaluation_model'"}), 400
        
        logger.info(f"🔍 Processing evaluation request (ID: {response_id}, model: {evaluation_model})")
        
        # Get stored response data
        response_data = get_gemini_response_by_id(response_id)
        if not response_data:
            return jsonify({"error": f"Response ID {response_id} not found"}), 404
        
        # Perform evaluation
        evaluation_result = evaluation_service.evaluate_responses(
            original_prompt=response_data['prompt'],
            gemini_without_search=response_data['response_without_search'],
            gemini_with_search=response_data['response_with_search'],
            evaluation_model=evaluation_model,
            custom_evaluation_prompt=custom_prompt
        )
        
        if "error" in evaluation_result:
            return jsonify(evaluation_result), 500
        
        # Save evaluation to database
        save_success = save_evaluation(
            response_id=response_id,
            evaluation_prompt=custom_prompt or "Default evaluation prompt",
            ollama_model=evaluation_model,
            used_without_search=evaluation_result['used_without_search'],
            used_with_search=evaluation_result['used_with_search'],
            result={"evaluation": evaluation_result['evaluation']}
        )
        
        if not save_success:
            logger.warning("⚠️ Failed to save evaluation to database")
        
        evaluation_result["success"] = True
        evaluation_result["response_id"] = response_id
        
        logger.info(f"✅ Evaluation completed successfully (ID: {response_id})")
        return jsonify(evaluation_result)
        
    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500


# =============================================================================
# DATA RETRIEVAL ROUTES
# =============================================================================

@api_bp.route('/api/history')
@api_bp.route('/api/gemini-history')
def get_history():
    """
    Get AI response history with status indicators.
    
    Returns:
        JSON with list of historical responses
    """
    try:
        history = get_gemini_responses_history()
        return jsonify({
            "success": True,
            "history": history,
            "total": len(history)
        })
    except Exception as e:
        error_msg = f"Failed to get history: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500


@api_bp.route('/api/response/<int:response_id>')
@api_bp.route('/api/gemini-response/<int:response_id>')
def get_response_details(response_id: int):
    """
    Get detailed information about a specific response.
    
    Args:
        response_id: Database ID of the response
        
    Returns:
        JSON with detailed response information
    """
    try:
        response_data = get_gemini_response_by_id(response_id)
        if not response_data:
            return jsonify({"error": f"Response ID {response_id} not found"}), 404
        
        # Get associated evaluations
        evaluations = get_evaluations_for_response(response_id)
        
        # Flatten response data to top level for frontend compatibility
        result = response_data.copy()
        result["success"] = True
        result["evaluations"] = evaluations
        
        return jsonify(result)
    except Exception as e:
        error_msg = f"Failed to get response details: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500


# =============================================================================
# MODEL AND SERVICE STATUS ROUTES
# =============================================================================

@api_bp.route('/api/models')
@api_bp.route('/api/available-models')
def get_available_models():
    """
    Get available models from both Gemini and Ollama services.
    
    Returns:
        JSON with available models from all services
    """
    try:
        result = {
            "success": True,
            "gemini_models": ["gemini-2.5-pro"],
            "ollama_models": [],
            "services_status": {
                "gemini": False,
                "ollama": False
            }
        }
        
        # Get Ollama models if service is available
        if ollama_service:
            try:
                result["ollama_models"] = ollama_service.get_available_models()
                result["services_status"]["ollama"] = True
            except Exception as e:
                logger.warning(f"⚠️ Failed to get Ollama models: {str(e)}")
        
        # Check Gemini service status
        if gemini_service:
            try:
                health_check = gemini_service.health_check()
                result["services_status"]["gemini"] = health_check.get("status") == "healthy"
            except Exception as e:
                logger.warning(f"⚠️ Failed to check Gemini status: {str(e)}")
        
        return jsonify(result)
    except Exception as e:
        error_msg = f"Failed to get available models: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500


@api_bp.route('/api/ollama-models')
def get_ollama_models():
    """
    Get available Ollama models specifically.
    
    Returns:
        JSON with Ollama models only
    """
    try:
        result = {
            "success": True,
            "models": [],
            "status": "offline"
        }
        
        # Get Ollama models if service is available
        if ollama_service:
            try:
                result["models"] = ollama_service.get_available_models()
                result["status"] = "online"
            except Exception as e:
                logger.warning(f"⚠️ Failed to get Ollama models: {str(e)}")
                result["status"] = "error"
                result["error"] = str(e)
        
        return jsonify(result)
    except Exception as e:
        error_msg = f"Failed to get Ollama models: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500


@api_bp.route('/api/health')
def health_check():
    """
    Comprehensive health check for all services.
    
    Returns:
        JSON with health status of all components
    """
    try:
        health_status = {
            "status": "healthy",
            "services": {},
            "database": {"status": "unknown"},
            "timestamp": int(time.time())
        }
        
        # Check Gemini service
        if gemini_service:
            try:
                gemini_health = gemini_service.health_check()
                health_status["services"]["gemini"] = gemini_health
            except Exception as e:
                health_status["services"]["gemini"] = {"status": "unhealthy", "error": str(e)}
        
        # Check Ollama service
        if ollama_service:
            try:
                ollama_health = ollama_service.health_check()
                health_status["services"]["ollama"] = ollama_health
            except Exception as e:
                health_status["services"]["ollama"] = {"status": "unhealthy", "error": str(e)}
        
        # Check database
        try:
            db_stats = get_database_stats()
            health_status["database"] = {"status": "healthy", "stats": db_stats}
        except Exception as e:
            health_status["database"] = {"status": "unhealthy", "error": str(e)}
        
        # Determine overall status
        service_statuses = [svc.get("status") for svc in health_status["services"].values()]
        if "unhealthy" in service_statuses or health_status["database"]["status"] == "unhealthy":
            health_status["status"] = "degraded"
        
        return jsonify(health_status)
        
    except Exception as e:
        error_msg = f"Health check failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return jsonify({"status": "unhealthy", "error": error_msg}), 500


# =============================================================================
# SYSTEM INFORMATION ROUTES
# =============================================================================

@api_bp.route('/api/stats')
def get_system_stats():
    """
    Get system statistics and database information.
    
    Returns:
        JSON with system statistics
    """
    try:
        stats = get_database_stats()
        
        # Add service information
        stats["services"] = {
            "gemini_available": gemini_service is not None,
            "ollama_available": ollama_service is not None,
            "evaluation_available": evaluation_service is not None
        }
        
        # Add configuration information (non-sensitive)
        stats["configuration"] = {
            "flask_env": Config.FLASK_ENV,
            "debug_mode": Config.FLASK_DEBUG,
            "ollama_base_url": Config.OLLAMA_BASE_URL,
            "gpu_monitoring": Config.ENABLE_GPU_MONITORING,
            "caching_enabled": Config.ENABLE_CACHING
        }
        
        return jsonify({
            "success": True,
            "stats": stats
        })
    except Exception as e:
        error_msg = f"Failed to get system stats: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500


# =============================================================================
# GPU MEMORY MANAGEMENT ROUTES
# =============================================================================

@api_bp.route('/api/loaded-models')
def get_loaded_models():
    """
    Get currently loaded models in GPU memory.
    
    Returns:
        JSON with currently loaded models information
    """
    try:
        result = {
            "success": True,
            "models": []
        }
        
        if ollama_service:
            loaded_models = ollama_service.get_loaded_models()
            result["models"] = loaded_models
        
        return jsonify(result)
    except Exception as e:
        error_msg = f"Failed to get loaded models: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500


@api_bp.route('/api/unload-model', methods=['POST'])
def unload_model():
    """
    Unload a specific model from GPU memory.
    
    Request JSON:
        {
            "model_name": "string"
        }
    
    Returns:
        JSON with unload result
    """
    try:
        if not ollama_service:
            return jsonify({"error": "Ollama service not initialized"}), 500
            
        data = request.get_json()
        if not data or 'model_name' not in data:
            return jsonify({"error": "Missing 'model_name' in request body"}), 400
        
        model_name = data['model_name']
        success = ollama_service.unload_model(model_name)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Successfully unloaded {model_name}"
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Failed to unload {model_name}"
            }), 500
            
    except Exception as e:
        error_msg = f"Failed to unload model: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500


@api_bp.route('/api/gpu-cleanup', methods=['POST'])
def gpu_cleanup():
    """
    Clean up all loaded models from GPU memory.
    
    Returns:
        JSON with cleanup results
    """
    try:
        if not ollama_service:
            return jsonify({"error": "Ollama service not initialized"}), 500
        
        results = ollama_service.cleanup_gpu_memory()
        
        return jsonify({
            "success": True,
            "message": "GPU cleanup completed",
            "results": results
        })
        
    except Exception as e:
        error_msg = f"GPU cleanup failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@api_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@api_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({"error": "Method not allowed"}), 405


@api_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"❌ Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


# Export Blueprint and initialization function
__all__ = ['api_bp', 'init_services']