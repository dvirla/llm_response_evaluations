# AI Model Response Evaluator - API Documentation

Complete API reference for the AI Model Response Evaluator platform.

## 📖 Overview

The API provides endpoints for:
- Multi-model AI querying (Gemini 2.5 Pro + Ollama models)
- Custom response evaluation  
- Response history management
- GPU memory management
- Model availability discovery

**Base URL**: `http://localhost:5001`
**Content-Type**: `application/json` for all POST requests

## 🔐 Authentication

Currently, the API does not require authentication. API keys are configured via environment variables on the server side.

## 📊 Response Format

All API responses follow a consistent format:

### Success Response
```json
{
  "success": true,
  "data": { ... },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Response  
```json
{
  "success": false,
  "error": "Error description",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🚀 Core Endpoints

### 1. Query AI Models

#### `POST /api/query-gemini`

Query any supported AI model with dual-mode comparison (with/without search).

**Request Body:**
```json
{
  "prompt": "What are the latest developments in quantum computing?",
  "model_name": "gemini-2.5-pro"
}
```

**Parameters:**
- `prompt` (string, required): The question/prompt to send to the model
- `model_name` (string, optional): Model identifier. Defaults to "gemini-2.5-pro"

**Supported Models:**
- `"gemini-2.5-pro"` - Google's Gemini 2.5 Pro
- `"llama3.1:70b"` - Meta's Llama 3.1 70B via Ollama  
- `"qwen2.5:14b"` - Qwen 2.5 14B via Ollama
- Any other Ollama model name

**Response:**
```json
{
  "response_id": 123,
  "prompt": "What are the latest developments in quantum computing?",
  "model_name": "gemini-2.5-pro",
  "response_without_search": {
    "thinking": "Let me think about quantum computing developments...",
    "response": "Recent quantum computing developments include...",
    "full_response": "Combined thinking and response text",
    "error": null
  },
  "response_with_search": {
    "thinking": "Let me search for recent quantum computing news...",
    "response": "Based on recent search results, quantum computing...",
    "full_response": "Combined thinking and response text", 
    "grounding_info": {
      "has_grounding": true,
      "search_queries": ["quantum computing 2024", "quantum developments"],
      "sources": [
        {
          "title": "Latest Quantum Computing Breakthroughs",
          "url": "https://example.com/article"
        }
      ]
    },
    "error": null
  },
  "saved_to_db": true
}
```

**Status Codes:**
- `200`: Success - Dual responses generated
- `400`: Bad Request - Missing or invalid prompt
- `500`: Internal Server Error - Model or API failure

---

### 2. Evaluate Responses

#### `POST /api/evaluate-response`

Evaluate stored responses using custom prompts and selected Ollama models.

**Request Body:**
```json
{
  "response_id": 123,
  "evaluation_prompt": "Compare these responses and rate their accuracy: {gemini_without_search_response} vs {gemini_with_search_response}",
  "ollama_model": "llama3.1:70b"
}
```

**Parameters:**
- `response_id` (integer, required): ID of the stored response pair to evaluate
- `evaluation_prompt` (string, required): Custom evaluation prompt with placeholders
- `ollama_model` (string, required): Ollama model to use for evaluation

**Template Placeholders:**
- `{original_prompt}` - User's original question
- `{gemini_without_search_thinking}` - Thinking process without search
- `{gemini_without_search_response}` - Final response without search  
- `{gemini_with_search_thinking}` - Thinking process with search
- `{gemini_with_search_response}` - Final response with search

**Response:**
```json
{
  "evaluation_result": {
    "evaluation": "The search-enabled response is significantly more accurate...",
    "model_used": "llama3.1:70b"
  },
  "evaluation_prompt_used": "Compare these responses and rate...",
  "placeholders_used": {
    "without_search_thinking": false,
    "without_search_response": true,
    "with_search_thinking": false, 
    "with_search_response": true,
    "original_prompt": false
  },
  "saved_to_db": true
}
```

**Status Codes:**
- `200`: Success - Evaluation completed
- `400`: Bad Request - Missing parameters or invalid response_id
- `404`: Not Found - Response ID doesn't exist
- `500`: Internal Server Error - Evaluation model failure

---

## 📚 Data Management Endpoints

### 3. Response History

#### `GET /api/gemini-history`

Retrieve historical AI model responses with search functionality.

**Query Parameters:**
- `search` (string, optional): Search term to filter responses by prompt content

**Example Request:**
```
GET /api/gemini-history?search=quantum
```

**Response:**
```json
{
  "history": [
    {
      "id": 123,
      "timestamp": "2024-01-15T10:30:00Z",
      "prompt": "What are the latest developments in quantum computing...",
      "full_prompt": "What are the latest developments in quantum computing and their implications?",
      "prompt_hash": "abc123def456",
      "model_name": "gemini-2.5-pro",
      "status": "success"
    }
  ],
  "total": 1
}
```

**Status Values:**
- `"success"` - Both responses completed successfully
- `"partial_failure"` - One response failed
- `"both_failed"` - Both responses failed

---

### 4. Individual Response Details

#### `GET /api/gemini-response/<response_id>`

Get detailed information for a specific response including evaluations.

**Path Parameters:**
- `response_id` (integer): ID of the response to retrieve

**Example Request:**
```
GET /api/gemini-response/123
```

**Response:**
```json
{
  "id": 123,
  "timestamp": "2024-01-15T10:30:00Z",
  "prompt": "What are the latest developments in quantum computing?",
  "prompt_hash": "abc123def456",
  "model_name": "gemini-2.5-pro",
  "response_without_search": {
    "response": "Quantum computing has seen several developments...",
    "thinking": "Let me consider the current state...",
    "error": null
  },
  "response_with_search": {
    "response": "Based on recent research and news...",
    "thinking": "I should search for the latest information...",
    "grounding_info": {
      "has_grounding": true,
      "search_queries": ["quantum computing 2024"],
      "sources": [...]
    },
    "error": null
  },
  "evaluations": [
    {
      "id": 456,
      "timestamp": "2024-01-15T10:35:00Z",
      "evaluation_prompt": "Compare these responses...",
      "ollama_model": "llama3.1:70b",
      "used_without_search": true,
      "used_with_search": true,
      "evaluation_result": "The search-enabled response...",
      "error_message": null
    }
  ]
}
```

---

## 🤖 Model Management Endpoints

### 5. Available Models

#### `GET /api/available-models`

Get all available AI models (Gemini + Ollama).

**Response:**
```json
{
  "models": [
    "gemini-2.5-pro",
    "llama3.1:70b", 
    "llama3.1:8b",
    "qwen2.5:14b",
    "deepseek-r1:70b"
  ]
}
```

---

#### `GET /api/ollama-models`

Get only Ollama models (for evaluation dropdown).

**Response:**
```json
{
  "models": [
    "llama3.1:70b",
    "llama3.1:8b", 
    "qwen2.5:14b",
    "deepseek-r1:70b"
  ]
}
```

---

## 🖥️ GPU Management Endpoints

### 6. GPU Memory Status

#### `GET /api/loaded-models`

Get currently loaded models and GPU memory status.

**Response:**
```json
{
  "success": true,
  "models": [
    {
      "name": "llama3.1:70b",
      "id": "sha256:abc123...",
      "size": "42 GB",
      "processor": "100% GPU", 
      "until": "4 minutes from now"
    },
    {
      "name": "qwen2.5:14b",
      "id": "sha256:def456...",
      "size": "8.5 GB",
      "processor": "100% GPU",
      "until": "2 minutes from now"  
    }
  ]
}
```

---

### 7. GPU Memory Cleanup

#### `POST /api/gpu-cleanup`

Unload all currently loaded models from GPU memory.

**Request Body:** (empty)
```json
{}
```

**Response:**
```json
{
  "success": true,
  "message": "Unloaded 2/2 models",
  "results": [
    {
      "model": "llama3.1:70b",
      "success": true,
      "message": "Model llama3.1:70b unloaded from GPU memory"
    },
    {
      "model": "qwen2.5:14b", 
      "success": true,
      "message": "Model qwen2.5:14b unloaded from GPU memory"
    }
  ]
}
```

---

#### `POST /api/unload-model`

Unload a specific model from GPU memory.

**Request Body:**
```json
{
  "model_name": "llama3.1:70b"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Model llama3.1:70b unloaded from GPU memory"
}
```

---

## 📄 Static Endpoints

### 8. Web Interface

#### `GET /`

Serve the main web application interface.

**Response:** HTML page with the complete web interface

---

## ⚠️ Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `MISSING_PROMPT` | Prompt parameter is required | Include 'prompt' in request body |
| `INVALID_MODEL` | Model name not recognized | Use `/api/available-models` to get valid models |
| `MODEL_UNAVAILABLE` | Requested model is not loaded/available | Check Ollama status or model availability |
| `API_KEY_MISSING` | Required API key not configured | Set environment variables |
| `RATE_LIMITED` | Too many requests | Wait before retrying |
| `GPU_MEMORY_FULL` | Insufficient GPU memory | Use `/api/gpu-cleanup` to free memory |
| `RESPONSE_NOT_FOUND` | Invalid response_id | Use valid ID from `/api/gemini-history` |
| `EVALUATION_FAILED` | Ollama evaluation error | Check Ollama service status |
| `SEARCH_UNAVAILABLE` | Search API not configured | Set TAVILY_API_KEY |

---

## 📊 Usage Examples

### Complete Workflow Example

```bash
# 1. Query a model
curl -X POST http://localhost:5001/api/query-gemini \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum entanglement",
    "model_name": "llama3.1:70b"
  }'

# Response: {"response_id": 123, ...}

# 2. Evaluate the responses  
curl -X POST http://localhost:5001/api/evaluate-response \
  -H "Content-Type: application/json" \
  -d '{
    "response_id": 123,
    "evaluation_prompt": "Rate the clarity of this explanation: {gemini_without_search_response}",
    "ollama_model": "qwen2.5:14b"
  }'

# 3. Check GPU memory
curl http://localhost:5001/api/loaded-models

# 4. Cleanup GPU memory
curl -X POST http://localhost:5001/api/gpu-cleanup
```

### Batch Processing Example

```bash
# Query multiple models for comparison
for model in "gemini-2.5-pro" "llama3.1:70b" "qwen2.5:14b"; do
  curl -X POST http://localhost:5001/api/query-gemini \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"What is machine learning?\", \"model_name\": \"$model\"}"
done
```

---

## 🔧 Configuration

### Environment Variables

The API behavior can be configured via environment variables:

```bash
# API timeouts
REQUEST_TIMEOUT=120
API_MAX_RETRIES=3

# GPU management  
OLLAMA_KEEP_ALIVE=5m
OLLAMA_MAX_LOADED_MODELS=3

# Rate limiting
ENABLE_RATE_LIMITING=false
RATE_LIMIT_PER_MINUTE=60
```

### Model Configuration

Models are automatically discovered from:
- **Gemini**: Configured via `GOOGLE_API_KEY`
- **Ollama**: Discovered via `ollama list` command

---

## 🚀 Performance Considerations

### Response Times
- **Gemini queries**: 5-30 seconds depending on complexity
- **Ollama queries**: 10-60 seconds depending on model size  
- **Evaluations**: 5-30 seconds depending on evaluation model

### Rate Limits
- **Gemini API**: Varies by API key tier
- **Tavily Search**: 1000 searches/month (free tier)
- **Application**: Configurable via environment variables

### GPU Memory
- Monitor with `/api/loaded-models`
- Cleanup with `/api/gpu-cleanup` when switching models
- Models auto-unload after `OLLAMA_KEEP_ALIVE` timeout

---

## 🛠️ Development & Testing

### Testing API Endpoints

```bash
# Health check
curl http://localhost:5001/

# Test model availability
curl http://localhost:5001/api/available-models

# Test basic query
curl -X POST http://localhost:5001/api/query-gemini \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "model_name": "gemini-2.5-pro"}'
```

### Monitoring

Monitor API performance via logs:
```bash
tail -f app.log | grep "API"
```

---

## 📞 Support

For API issues:
1. Check application logs for detailed error messages
2. Verify environment variables are set correctly  
3. Test individual components (Gemini API, Ollama, Tavily)
4. Create GitHub issue with request/response examples

---

**API Version**: v1.0  
**Last Updated**: January 2024