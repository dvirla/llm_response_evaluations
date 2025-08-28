#!/usr/bin/env python3
"""
Database Models Module - AI Model Response Evaluator

Pure database operations and schema management for the AI Model Response Evaluator.
Handles SQLite database operations, schema creation, and all CRUD operations
for responses, evaluations, and batch jobs.

Author: dvirla
License: MIT
"""

import sqlite3
import json
import hashlib
from typing import Dict, Any, List, Optional, Union
from config import logger, DATABASE_PATH

# =============================================================================
# DATABASE CONNECTION AND INITIALIZATION
# =============================================================================

def get_db_connection() -> sqlite3.Connection:
    """
    Create and configure a database connection with Row factory.
    
    Returns:
        sqlite3.Connection: Database connection with Row factory enabled
                           for dictionary-like access to query results
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """
    Initialize the SQLite database with enhanced schema for separated workflow.
    
    Creates three main tables:
    1. gemini_responses: Stores AI model responses from both search modes
    2. evaluations: Stores user-controlled evaluations of responses  
    3. batch_jobs: Tracks batch processing operations
    
    Includes automatic migration to add model_name column to existing installations.
    
    Raises:
        sqlite3.Error: If database creation or migration fails
    """
    with sqlite3.connect(DATABASE_PATH) as conn:
        # Create gemini_responses table - stores responses from both modes
        conn.execute('''
            CREATE TABLE IF NOT EXISTS gemini_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                prompt TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                model_name VARCHAR(100) DEFAULT 'gemini-2.5-pro',
                response_without_search TEXT NOT NULL,
                thinking_without_search TEXT,
                response_with_search TEXT NOT NULL,
                thinking_with_search TEXT,
                grounding_info TEXT,
                error_without_search TEXT,
                error_with_search TEXT
            )
        ''')
        
        # Add model_name column to existing table if it doesn't exist (migration)
        try:
            conn.execute('ALTER TABLE gemini_responses ADD COLUMN model_name VARCHAR(100) DEFAULT "gemini-2.5-pro"')
            logger.info("📊 Added model_name column to existing gemini_responses table")
        except sqlite3.OperationalError:
            # Column already exists, ignore
            pass
        
        # Create evaluations table - stores user-controlled evaluations
        conn.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                gemini_response_id INTEGER NOT NULL,
                evaluation_prompt TEXT NOT NULL,
                ollama_model TEXT NOT NULL,
                used_without_search BOOLEAN DEFAULT FALSE,
                used_with_search BOOLEAN DEFAULT FALSE,
                evaluation_result TEXT,
                error_message TEXT,
                FOREIGN KEY (gemini_response_id) REFERENCES gemini_responses (id)
            )
        ''')
        
        # Create batch_jobs table for batch processing
        conn.execute('''
            CREATE TABLE IF NOT EXISTS batch_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL DEFAULT 'pending',
                total_questions INTEGER NOT NULL,
                completed_questions INTEGER DEFAULT 0,
                failed_questions INTEGER DEFAULT 0,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                file_path TEXT NOT NULL,
                error_message TEXT,
                processed_questions TEXT DEFAULT '[]',
                retry_count INTEGER DEFAULT 0,
                last_error_time DATETIME
            )
        ''')
        
        conn.commit()
    
    logger.info("📊 Database initialized with separated workflow schema")

# =============================================================================
# RESPONSE STORAGE OPERATIONS
# =============================================================================

def save_gemini_responses(prompt: str, result_no_search: Dict[str, Any], 
                         result_with_search: Dict[str, Any], model_name: str = 'gemini-2.5-pro') -> Optional[int]:
    """
    Save AI model responses to the database with comprehensive error handling.
    
    Stores both search and non-search responses along with thinking processes,
    grounding information, and any errors encountered during generation.
    
    Args:
        prompt: The original user prompt/query
        result_no_search: Response data from model without search capabilities
        result_with_search: Response data from model with search capabilities
        model_name: Name of the AI model used (default: 'gemini-2.5-pro')
        
    Returns:
        Optional[int]: Database ID of saved response record, None if save failed
        
    Note:
        Grounding information is stored as JSON string for complex data structures.
        MD5 hash of prompt is stored for quick duplicate detection.
    """
    try:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        # Convert grounding info to JSON string if exists
        grounding_info_str = None
        if result_with_search.get('grounding_info'):
            grounding_info_str = json.dumps(result_with_search['grounding_info'])
        
        with get_db_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO gemini_responses 
                (prompt, prompt_hash, model_name, response_without_search, thinking_without_search,
                 response_with_search, thinking_with_search, grounding_info,
                 error_without_search, error_with_search)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prompt, prompt_hash, model_name,
                result_no_search.get('response', ''), result_no_search.get('thinking', ''),
                result_with_search.get('response', ''), result_with_search.get('thinking', ''),
                grounding_info_str,
                result_no_search.get('error'), result_with_search.get('error')
            ))
            response_id = cursor.lastrowid
            conn.commit()
        
        logger.info(f"💾 Saved {model_name} responses (ID: {response_id}, hash: {prompt_hash[:8]})")
        return response_id
    except Exception as e:
        logger.error(f"❌ Failed to save {model_name} responses: {str(e)}")
        return None

def save_evaluation(response_id: int, evaluation_prompt: str, ollama_model: str, 
                   used_without_search: bool, used_with_search: bool, result: Dict[str, Any]) -> bool:
    """
    Save evaluation results to the database with metadata tracking.
    
    Links evaluation results to their corresponding AI model responses while tracking
    which response modes were used in the evaluation process.
    
    Args:
        response_id: Foreign key to gemini_responses table
        evaluation_prompt: Custom evaluation prompt used
        ollama_model: Name of the Ollama model used for evaluation
        used_without_search: Whether non-search response was included
        used_with_search: Whether search-enabled response was included
        result: Evaluation result dictionary containing 'evaluation' and/or 'error' keys
        
    Returns:
        bool: True if evaluation was successfully saved, False otherwise
    """
    try:
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO evaluations 
                (gemini_response_id, evaluation_prompt, ollama_model, 
                 used_without_search, used_with_search, evaluation_result, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                response_id, evaluation_prompt, ollama_model,
                used_without_search, used_with_search,
                result.get('evaluation'), result.get('error')
            ))
            conn.commit()
        
        logger.info(f"💾 Saved evaluation for response ID {response_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save evaluation: {str(e)}")
        return False

# =============================================================================
# RESPONSE RETRIEVAL OPERATIONS
# =============================================================================

def get_gemini_responses_history() -> List[Dict[str, Any]]:
    """
    Get all AI model response history with status indicators.
    
    Returns:
        List[Dict]: List of response history items with metadata including:
        - id, timestamp, prompt, model_name
        - status: 'success', 'partial_failure', or 'both_failed'
    """
    try:
        with get_db_connection() as conn:
            rows = conn.execute('''
                SELECT id, timestamp, prompt, prompt_hash, model_name,
                       (CASE WHEN error_without_search IS NULL AND error_with_search IS NULL THEN 'success'
                             WHEN error_without_search IS NOT NULL AND error_with_search IS NOT NULL THEN 'both_failed'
                             ELSE 'partial_failure' END) as status
                FROM gemini_responses 
                ORDER BY timestamp DESC
            ''').fetchall()
        
        history = []
        for row in rows:
            history.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'prompt': row['prompt'][:100] + ('...' if len(row['prompt']) > 100 else ''),
                'full_prompt': row['prompt'],
                'prompt_hash': row['prompt_hash'],
                'model_name': row['model_name'] or 'gemini-2.5-pro',  # Handle legacy records
                'status': row['status']
            })
        
        return history
    except Exception as e:
        logger.error(f"❌ Failed to get history: {str(e)}")
        return []

def get_gemini_response_by_id(response_id: int) -> Optional[Dict[str, Any]]:
    """
    Get specific AI model response by ID with full details.
    
    Args:
        response_id: Database ID of the response to retrieve
        
    Returns:
        Optional[Dict]: Response data with both modes and metadata, None if not found
    """
    try:
        with get_db_connection() as conn:
            row = conn.execute('''
                SELECT * FROM gemini_responses WHERE id = ?
            ''', (response_id,)).fetchone()
        
        if row:
            # Parse grounding info if exists
            grounding_info = None
            if row['grounding_info']:
                try:
                    grounding_info = json.loads(row['grounding_info'])
                except:
                    grounding_info = None
            
            return {
                'id': row['id'],
                'timestamp': row['timestamp'],
                'prompt': row['prompt'],
                'prompt_hash': row['prompt_hash'],
                'model_name': row['model_name'] or 'gemini-2.5-pro',  # Handle legacy records
                'response_without_search': {
                    'response': row['response_without_search'],
                    'thinking': row['thinking_without_search'],
                    'error': row['error_without_search']
                },
                'response_with_search': {
                    'response': row['response_with_search'],
                    'thinking': row['thinking_with_search'],
                    'grounding_info': grounding_info,
                    'error': row['error_with_search']
                }
            }
        return None
    except Exception as e:
        logger.error(f"❌ Failed to get response {response_id}: {str(e)}")
        return None

def get_evaluations_for_response(response_id: int) -> List[Dict[str, Any]]:
    """
    Get all evaluations for a specific AI model response.
    
    Args:
        response_id: Database ID of the response to get evaluations for
        
    Returns:
        List[Dict]: List of evaluation records with metadata
    """
    try:
        with get_db_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM evaluations 
                WHERE gemini_response_id = ? 
                ORDER BY timestamp DESC
            ''', (response_id,)).fetchall()
        
        evaluations = []
        for row in rows:
            evaluations.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'evaluation_prompt': row['evaluation_prompt'],
                'ollama_model': row['ollama_model'],
                'used_without_search': bool(row['used_without_search']),
                'used_with_search': bool(row['used_with_search']),
                'evaluation_result': row['evaluation_result'],
                'error_message': row['error_message']
            })
        
        return evaluations
    except Exception as e:
        logger.error(f"❌ Failed to get evaluations for response {response_id}: {str(e)}")
        return []

# =============================================================================
# BATCH PROCESSING OPERATIONS (Future Feature)
# =============================================================================

def create_batch_job(file_path: str, total_questions: int) -> Optional[int]:
    """
    Create a new batch processing job record.
    
    Args:
        file_path: Path to the input file
        total_questions: Number of questions to process
        
    Returns:
        Optional[int]: Batch job ID if created successfully, None otherwise
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO batch_jobs (file_path, total_questions)
                VALUES (?, ?)
            ''', (file_path, total_questions))
            job_id = cursor.lastrowid
            conn.commit()
        
        logger.info(f"📋 Created batch job {job_id} for {total_questions} questions")
        return job_id
    except Exception as e:
        logger.error(f"❌ Failed to create batch job: {str(e)}")
        return None

def update_batch_job_progress(job_id: int, completed: int, failed: int, 
                             processed_questions: List[str]) -> bool:
    """
    Update batch job progress and status.
    
    Args:
        job_id: Batch job ID
        completed: Number of completed questions
        failed: Number of failed questions  
        processed_questions: List of processed question IDs
        
    Returns:
        bool: True if update successful, False otherwise
    """
    try:
        processed_json = json.dumps(processed_questions)
        total_processed = completed + failed
        
        with get_db_connection() as conn:
            conn.execute('''
                UPDATE batch_jobs 
                SET completed_questions = ?, failed_questions = ?, 
                    processed_questions = ?
                WHERE id = ?
            ''', (completed, failed, processed_json, job_id))
            conn.commit()
        
        logger.info(f"📋 Updated batch job {job_id}: {total_processed} processed")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to update batch job {job_id}: {str(e)}")
        return False

# =============================================================================
# DATABASE MAINTENANCE
# =============================================================================

def get_database_stats() -> Dict[str, Any]:
    """
    Get database statistics for monitoring and maintenance.
    
    Returns:
        Dict: Database statistics including table counts and sizes
    """
    try:
        with get_db_connection() as conn:
            # Get table counts
            response_count = conn.execute('SELECT COUNT(*) FROM gemini_responses').fetchone()[0]
            evaluation_count = conn.execute('SELECT COUNT(*) FROM evaluations').fetchone()[0]
            batch_job_count = conn.execute('SELECT COUNT(*) FROM batch_jobs').fetchone()[0]
            
            # Get recent activity
            recent_responses = conn.execute('''
                SELECT COUNT(*) FROM gemini_responses 
                WHERE timestamp > datetime('now', '-7 days')
            ''').fetchone()[0]
        
        return {
            'total_responses': response_count,
            'total_evaluations': evaluation_count,
            'total_batch_jobs': batch_job_count,
            'recent_responses_7d': recent_responses,
            'database_path': DATABASE_PATH
        }
    except Exception as e:
        logger.error(f"❌ Failed to get database stats: {str(e)}")
        return {}

# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Initialize database on module import
try:
    init_db()
except Exception as e:
    logger.error(f"❌ Failed to initialize database: {str(e)}")
    raise

# Export main functions
__all__ = [
    'get_db_connection',
    'init_db',
    'save_gemini_responses',
    'save_evaluation',
    'get_gemini_responses_history',
    'get_gemini_response_by_id',
    'get_evaluations_for_response',
    'create_batch_job',
    'update_batch_job_progress',
    'get_database_stats'
]