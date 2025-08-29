# AI Model Response Evaluator - System Architecture

This document outlines the system architecture, design decisions, and technical implementation details of the AI Model Response Evaluator platform.

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│  Web Browser (HTML/CSS/JavaScript)                             │
│  • Responsive design with mobile support                       │
│  • Real-time status updates and progress tracking              │
│  • Dynamic model selection and GPU memory controls             │
└─────────────────┬───────────────────────────────────────────────┘
                  │ HTTP/HTTPS
                  │ REST API Calls
┌─────────────────▼───────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  Flask Web Framework (Python 3.7+)                            │
│  • RESTful API endpoints                                       │
│  • Request validation and error handling                       │
│  • Session management and state tracking                       │
│  • Async task coordination                                     │
└─────────────┬───────────┬───────────────────────────────────────┘
              │           │
      ┌───────▼────┐ ┌────▼─────────┐
      │  BUSINESS  │ │  DATA ACCESS │
      │   LOGIC    │ │    LAYER     │
      └─────┬──────┘ └────┬─────────┘
            │             │
┌───────────▼─────────────▼───────────────────────────────────────┐
│                     SERVICE LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  GeminiQuerier   │  │   Database   │  │ PydanticAI       │  │
│  │                  │  │   Manager    │  │ Wrapper          │  │
│  │ • Dual-mode      │  │              │  │                  │  │
│  │   querying       │  │ • SQLite     │  │ • Ollama         │  │
│  │ • Response       │  │   operations │  │   integration    │  │
│  │   parsing        │  │ • History    │  │ • GPU management │  │
│  │ • Error handling │  │   tracking   │  │ • Search tools   │  │
│  └──────────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────┬───────────────┬───────────────┬─────────────────┘
              │               │               │
┌─────────────▼─────┐ ┌───────▼──────┐ ┌──────▼────────────────────┐
│  EXTERNAL APIs    │ │   DATABASE   │ │    LOCAL SERVICES         │
├───────────────────┤ ├──────────────┤ ├───────────────────────────┤
│ • Google Gemini   │ │  SQLite DB   │ │ • Ollama Server           │
│   2.5 Pro API     │ │              │ │   (http://localhost:11434)│
│ • Tavily Search   │ │ Tables:      │ │ • Model Management        │
│   API             │ │ - responses  │ │ • GPU Memory Control      │
│                   │ │ - evaluations│ │ • OpenAI Compatible API   │
│                   │ │ - batch_jobs │ │                           │
└───────────────────┘ └──────────────┘ └───────────────────────────┘
```

## 🧩 Component Architecture

### 1. Frontend Layer (Web Interface)

**Technology Stack:**
- **HTML5**: Semantic markup with accessibility considerations
- **CSS3**: Responsive design with Flexbox/Grid layouts
- **Vanilla JavaScript**: No external frameworks for simplicity
- **WebSocket**: (Future) Real-time updates for long-running operations

**Key Components:**
```javascript
// Main application state
- currentResponseId: Tracks active response pair
- currentPrompt: Current user input
- currentHistory: Cached response history

// UI Controllers
- ModelSelector: Dynamic model dropdown population
- ResponseRenderer: Formats and displays AI responses  
- HistoryManager: Search and load previous responses
- GPUController: Memory monitoring and cleanup
```

**Design Patterns:**
- **Module Pattern**: Isolated functionality in namespace objects
- **Observer Pattern**: Event-driven UI updates
- **Command Pattern**: GPU management operations

### 2. Backend Layer (Flask Application)

**Technology Stack:**
- **Flask 2.3+**: Lightweight WSGI web framework
- **Python 3.7+**: Core application language
- **SQLite**: Embedded database for simplicity
- **dotenv**: Environment variable management

**Core Modules:**

#### `app.py` - Main Application
```python
class GeminiQuerier:
    """Unified AI model interface"""
    - query_gemini_without_search()
    - query_gemini_with_search() 
    - query_model_without_search()  # Unified interface
    - query_model_with_search()     # Unified interface
    - evaluate_with_ollama()
    - get_available_models()
    
# API Routes
- /                              # Web interface
- /api/query-gemini             # Dual-mode querying
- /api/evaluate-response        # Custom evaluations
- /api/gemini-history          # Response history
- /api/available-models        # Model discovery
- /api/gpu-cleanup             # Memory management
```

#### `pydantic_ai_wrapper.py` - Ollama Integration
```python
class PydanticAIOllamaWrapper:
    """Advanced Ollama integration with GPU management"""
    - query_ollama_without_search()
    - query_ollama_with_search()
    - unload_model()              # Individual model cleanup
    - unload_all_models()         # Batch cleanup
    - get_loaded_models()         # Status monitoring

class TavilySearchTool:
    """AI-optimized web search integration"""
    - search()                    # Execute search queries
    - format_results()            # Structure for LLM consumption
```

### 3. Data Layer

#### Database Schema Design

```sql
-- Core response storage
CREATE TABLE gemini_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    prompt TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,              -- MD5 for deduplication
    model_name VARCHAR(100),                -- Model attribution
    response_without_search TEXT NOT NULL,
    thinking_without_search TEXT,           -- Reasoning transparency
    response_with_search TEXT NOT NULL,
    thinking_with_search TEXT,
    grounding_info TEXT,                    -- JSON search metadata
    error_without_search TEXT,              -- Error tracking
    error_with_search TEXT
);

-- Evaluation tracking
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    gemini_response_id INTEGER NOT NULL,   -- Foreign key
    evaluation_prompt TEXT NOT NULL,       -- Custom prompt used
    ollama_model TEXT NOT NULL,           -- Evaluation model
    used_without_search BOOLEAN,          -- Response mode tracking
    used_with_search BOOLEAN,
    evaluation_result TEXT,
    error_message TEXT,
    FOREIGN KEY (gemini_response_id) REFERENCES gemini_responses (id)
);
```

**Design Decisions:**
- **SQLite**: Simplicity for single-user applications
- **JSON Storage**: Flexible schema for grounding_info
- **Error Tracking**: Separate error columns for debugging
- **Foreign Keys**: Referential integrity between responses and evaluations

### 4. Integration Layer

#### External API Integration

**Google Gemini 2.5 Pro**
```python
# Configuration
- API Key: GOOGLE_API_KEY environment variable
- Endpoint: Google GenAI REST API
- Features: Thinking process, search tools, structured responses

# Error Handling
- Exponential backoff (5s, 15s, 45s)
- Response validation
- Partial failure recovery
```

**Ollama Integration**  
```python
# Configuration
- Base URL: http://localhost:11434/v1 (OpenAI-compatible)
- Models: Dynamic discovery via `ollama list`
- Memory: Native keep_alive parameter management

# Advanced Features
- GPU memory monitoring via `ollama ps`
- Individual model unloading
- Batch cleanup operations
```

**Tavily Search API**
```python
# Configuration  
- API Key: TAVILY_API_KEY environment variable
- Endpoint: https://api.tavily.com/search
- Features: AI-optimized results, clean content extraction

# Integration
- Seamless pydantic_ai tool integration
- Structured result formatting
- Error fallback handling
```

## 🔄 Data Flow Architecture

### 1. Query Processing Flow

```
User Input (Prompt + Model)
    ↓
Input Validation & Sanitization  
    ↓
Model Route Determination
    ├─ Gemini 2.5 Pro → Google GenAI API
    └─ Ollama Models → PydanticAI Wrapper
    ↓
Parallel Execution:
    ├─ Query WITHOUT Search
    └─ Query WITH Search (+ Tavily/Google Search)
    ↓
Response Processing & Parsing
    ├─ Extract thinking process
    ├─ Extract final response  
    └─ Extract grounding information
    ↓
Database Storage (with model attribution)
    ↓
Return Structured Response to Frontend
```

### 2. Evaluation Processing Flow

```
User Evaluation Request
    ├─ Response ID
    ├─ Custom Evaluation Prompt  
    └─ Selected Ollama Model
    ↓
Response Retrieval from Database
    ↓
Template Processing
    ├─ Replace {original_prompt}
    ├─ Replace {gemini_without_search_*}
    └─ Replace {gemini_with_search_*}
    ↓
Ollama Model Execution
    ↓
Evaluation Result Processing
    ↓
Database Storage (linked to original response)
    ↓
Return Evaluation to Frontend
```

### 3. GPU Management Flow

```
GPU Status Request
    ↓
Execute `ollama ps` Command
    ↓
Parse Model Information
    ├─ Model Names
    ├─ Memory Usage
    ├─ Processor Assignment
    └─ Timeout Information
    ↓
Format for Frontend Display
    ↓
Enable Individual/Batch Unload Operations
    ↓
Execute Unload via Ollama API
    ├─ Send keep_alive=0 request
    └─ Monitor success/failure
    ↓
Update Frontend Status
```

## 🚀 Performance Architecture

### 1. Caching Strategy

**Application-Level Caching:**
- Model discovery results (refreshed periodically)
- Response history (cached with invalidation)
- GPU status information (short-term cache)

**Database Optimization:**
- Indexes on frequently queried columns (timestamp, model_name)
- Prompt hash for duplicate detection
- Connection pooling for concurrent requests

### 2. Resource Management

**Memory Management:**
```python
# Ollama GPU Memory
- Automatic model unloading after timeout
- Manual cleanup via API endpoints
- Memory threshold monitoring

# Application Memory  
- Lazy loading of large datasets
- Streaming for large responses
- Garbage collection optimization
```

**Concurrency Handling:**
- Async support for long-running operations
- Thread-safe database operations
- Request queuing for resource-intensive operations

### 3. Scalability Considerations

**Horizontal Scaling:**
- Stateless application design
- External database option (PostgreSQL/MySQL)
- Load balancer support

**Vertical Scaling:**
- Multi-GPU support for Ollama
- Memory-mapped file caching
- CPU-intensive task optimization

## 🛡️ Security Architecture

### 1. API Security

**Input Validation:**
```python
- Prompt sanitization (XSS prevention)
- Parameter validation (type checking)
- Request size limits
- Rate limiting (configurable)
```

**Authentication & Authorization:**
```python
# Current: Server-side API key management
# Future: JWT tokens, OAuth2, role-based access
```

### 2. Data Security

**Environment Variables:**
- Sensitive API keys stored in .env
- No hardcoded secrets in source code
- Production-ready secret management support

**Database Security:**
- SQLite file permissions
- Connection string protection
- Query parameterization (SQL injection prevention)

### 3. Network Security

**API Endpoints:**
- HTTPS support (configurable)
- CORS policy configuration
- Request/response sanitization

## 🔧 Configuration Architecture

### 1. Environment-Based Configuration

```python
# Required Configuration
GOOGLE_API_KEY=           # Gemini API access
TAVILY_API_KEY=           # Search functionality

# Optional Configuration  
OLLAMA_BASE_URL=          # Ollama server location
FLASK_ENV=                # Development/Production
LOG_LEVEL=                # Logging verbosity
```

### 2. Feature Flags

```python
# Configurable Features
ENABLE_GPU_MONITORING=    # GPU memory tracking
ENABLE_CACHING=           # Response caching
ENABLE_RATE_LIMITING=     # API rate limits
ENABLE_CORS=              # Cross-origin requests
```

## 📊 Monitoring & Observability

### 1. Logging Architecture

**Structured Logging:**
```python
- Request/Response logging
- Error tracking with stack traces
- Performance metrics (response times)
- GPU memory usage tracking
```

**Log Levels:**
- DEBUG: Detailed troubleshooting information
- INFO: General application flow
- WARNING: Recoverable issues
- ERROR: Error conditions requiring attention

### 2. Metrics Collection

**Application Metrics:**
- Request count and latency
- Model usage statistics
- Error rates by endpoint
- GPU memory utilization

**Business Metrics:**
- Response quality scores
- Model performance comparisons
- User engagement patterns

## 🔄 Future Architecture Considerations

### 1. Microservices Migration

**Potential Service Breakdown:**
```
├── API Gateway Service
├── Model Management Service  
├── Evaluation Service
├── Search Integration Service
├── Database Service
└── GPU Management Service
```

### 2. Advanced Features

**Batch Processing:**
- Queue-based architecture
- Background job processing
- Progress tracking and notifications

**Real-time Features:**
- WebSocket integration
- Live response streaming
- Real-time collaboration

**Machine Learning Pipeline:**
- Response quality prediction
- Automated evaluation scoring
- Model recommendation system

---

## 🏗️ Development Guidelines

### 1. Code Organization

```
app.py                     # Main Flask application
├── Database functions     # Data access layer
├── GeminiQuerier class   # Business logic
└── API endpoints         # Presentation layer

pydantic_ai_wrapper.py    # External integrations
├── TavilySearchTool      # Search functionality  
└── PydanticAIOllamaWrapper # Ollama + GPU management

templates/new_index.html   # Frontend interface
├── HTML structure        # Semantic markup
├── CSS styling          # Responsive design
└── JavaScript logic     # Interactive functionality
```

### 2. Testing Strategy

**Unit Testing:**
- Individual function testing
- Mock external API calls
- Database operation testing

**Integration Testing:**  
- End-to-end API workflows
- Database migration testing
- External service integration

**Performance Testing:**
- Load testing for concurrent users
- Memory usage under load
- GPU memory management efficiency

---

This architecture provides a solid foundation for the AI Model Response Evaluator while maintaining flexibility for future enhancements and scalability requirements.