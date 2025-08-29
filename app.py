#!/usr/bin/env python3
"""
Main Application Entry Point - AI Model Response Evaluator

Minimal Flask application entry point that orchestrates the modular services
for AI response evaluation. Handles service initialization, error management,
and application lifecycle.

Author: dvirla
License: MIT
"""

import time
import signal
import atexit
from flask import Flask
from werkzeug.exceptions import HTTPException

# Import configuration and logging
from config import Config, logger

# Import services
from gemini_service import GeminiService
from ollama_service import OllamaService
from evaluation_service import EvaluationService

# Import routes
from routes import api_bp, init_services


class AIResponseEvaluatorApp:
    """
    Main application class for the AI Model Response Evaluator.
    
    Handles service initialization, Flask app configuration, and
    graceful shutdown procedures for optimal resource management.
    
    Attributes:
        app: Flask application instance
        gemini_service: GeminiService instance
        ollama_service: OllamaService instance  
        evaluation_service: EvaluationService instance
    """
    
    def __init__(self):
        """Initialize the application with service orchestration."""
        logger.info("🚀 Initializing AI Model Response Evaluator...")
        
        # Create Flask app
        self.app = self._create_flask_app()
        
        # Initialize services
        self._initialize_services()
        
        # Register routes
        self._register_routes()
        
        # Setup graceful shutdown
        self._setup_shutdown_handlers()
        
        logger.info("✅ AI Model Response Evaluator initialized successfully")
    
    def _create_flask_app(self) -> Flask:
        """
        Create and configure Flask application instance.
        
        Returns:
            Flask: Configured Flask application
        """
        app = Flask(__name__)
        
        # Basic Flask configuration
        app.config['SECRET_KEY'] = Config.FLASK_SECRET_KEY
        app.config['DEBUG'] = Config.FLASK_DEBUG
        
        # Configure JSON handling
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
        app.config['JSON_SORT_KEYS'] = False
        
        # Configure request handling
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size
        
        # Setup CORS if enabled
        if Config.ENABLE_CORS:
            self._setup_cors(app)
        
        # Setup error handlers
        self._setup_error_handlers(app)
        
        logger.info(f"🌐 Flask app created (debug: {Config.FLASK_DEBUG})")
        return app
    
    def _setup_cors(self, app: Flask) -> None:
        """
        Setup CORS configuration for the Flask app.
        
        Args:
            app: Flask application instance
        """
        try:
            from flask_cors import CORS
            CORS(app, origins=Config.get_cors_origins())
            logger.info("🌍 CORS enabled for specified origins")
        except ImportError:
            logger.warning("⚠️ flask-cors not installed, CORS not enabled")
    
    def _setup_error_handlers(self, app: Flask) -> None:
        """
        Setup global error handlers for the Flask app.
        
        Args:
            app: Flask application instance
        """
        @app.errorhandler(HTTPException)
        def handle_http_exception(e):
            logger.warning(f"⚠️ HTTP {e.code}: {e.description}")
            return {"error": e.description}, e.code
        
        @app.errorhandler(Exception)
        def handle_general_exception(e):
            logger.error(f"❌ Unhandled exception: {str(e)}")
            return {"error": "Internal server error"}, 500
    
    def _initialize_services(self) -> None:
        """
        Initialize all required services with proper error handling.
        
        Raises:
            SystemExit: If critical services fail to initialize
        """
        try:
            # Initialize Gemini service
            logger.info("🔧 Initializing Gemini service...")
            self.gemini_service = GeminiService()
            
            # Initialize Ollama service
            logger.info("🔧 Initializing Ollama service...")
            self.ollama_service = OllamaService()
            
            # Initialize evaluation service
            logger.info("🔧 Initializing evaluation service...")
            self.evaluation_service = EvaluationService(self.ollama_service)
            
            logger.info("✅ All services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Critical service initialization failed: {str(e)}")
            logger.error("💡 Check your configuration and API keys")
            raise SystemExit(1)
    
    def _register_routes(self) -> None:
        """Register application routes and initialize route services."""
        try:
            # Initialize services for routes module
            init_services(
                gemini_svc=self.gemini_service,
                ollama_svc=self.ollama_service,
                eval_svc=self.evaluation_service
            )
            
            # Register the API blueprint
            self.app.register_blueprint(api_bp)
            
            logger.info("🛣️ Routes registered successfully")
            
        except Exception as e:
            logger.error(f"❌ Route registration failed: {str(e)}")
            raise SystemExit(1)
    
    def _setup_shutdown_handlers(self) -> None:
        """Setup graceful shutdown handlers for proper resource cleanup."""
        def cleanup_resources():
            """Clean up application resources."""
            try:
                logger.info("🧹 Starting graceful shutdown...")
                
                # Cleanup services
                if hasattr(self, 'evaluation_service'):
                    self.evaluation_service.cleanup_resources()
                
                if hasattr(self, 'ollama_service'):
                    self.ollama_service.cleanup_all_agents()
                
                logger.info("✅ Graceful shutdown completed")
            except Exception as e:
                logger.error(f"❌ Error during shutdown: {str(e)}")
        
        # Register cleanup for normal exit
        atexit.register(cleanup_resources)
        
        # Register cleanup for signal termination
        def signal_handler(signum, frame):
            logger.info(f"📡 Received signal {signum}, shutting down...")
            cleanup_resources()
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run(self, host: str = None, port: int = None, debug: bool = None) -> None:
        """
        Run the Flask application with specified or configured parameters.
        
        Args:
            host: Host to bind to (uses config default if None)
            port: Port to bind to (uses config default if None)
            debug: Debug mode (uses config default if None)
        """
        host = host or Config.FLASK_HOST
        port = port or Config.FLASK_PORT
        debug = debug if debug is not None else Config.FLASK_DEBUG
        
        logger.info(f"🌐 Starting server on {host}:{port} (debug: {debug})")
        
        try:
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                use_reloader=False  # Disable reloader to avoid double initialization
            )
        except Exception as e:
            logger.error(f"❌ Failed to start server: {str(e)}")
            raise SystemExit(1)


def create_app() -> Flask:
    """
    Factory function to create Flask application instance.
    
    Used for deployment scenarios where the app instance needs to be
    created without running the development server.
    
    Returns:
        Flask: Configured Flask application instance
    """
    app_instance = AIResponseEvaluatorApp()
    return app_instance.app


def main():
    """
    Main entry point for the application.
    
    Handles command-line execution and development server startup.
    """
    try:
        # Create and run application
        app_instance = AIResponseEvaluatorApp()
        app_instance.run()
        
    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")
    except SystemExit:
        # Re-raise SystemExit to preserve exit codes
        raise
    except Exception as e:
        logger.error(f"❌ Application failed to start: {str(e)}")
        raise SystemExit(1)


if __name__ == '__main__':
    main()