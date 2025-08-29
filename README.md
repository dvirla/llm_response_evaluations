# AI Model Response Evaluator

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive multi-model AI comparison platform for evaluating responses from Gemini 2.5 Pro and Ollama models. Features dual-mode querying (with/without search), custom evaluation prompts, GPU memory management, and historical response tracking.

## 🚀 Features

### Multi-Model Support
- **Gemini 2.5 Pro**: Google's latest model with thinking process visibility
- **Ollama Models**: Any locally-hosted open-source models (llama3.1, qwen2.5, etc.)
- **Dynamic Model Selection**: Switch between models seamlessly

### Dual-Mode Querying  
- **Without Search**: Pure model knowledge and reasoning
- **With Search**: Enhanced with real-time web search (Gemini uses Google Search, Ollama uses Tavily)
- **Side-by-Side Comparison**: View both responses simultaneously

### Advanced GPU Management
- **Memory Monitoring**: Real-time view of loaded models and VRAM usage
- **Smart Unloading**: Individual or batch model unloading from GPU memory
- **Efficient Switching**: Free up resources when switching between models

### Customizable Evaluation
- **Template System**: Use placeholders for dynamic evaluation prompts
- **Multiple Models**: Choose any Ollama model for evaluation
- **Response Selection**: Evaluate specific response modes (with/without search)

### Data Management
- **Response History**: Track all queries and responses with timestamps
- **Model Attribution**: See which model generated each response
- **Evaluation Tracking**: Store multiple evaluations per response pair
- **Search Functionality**: Find past responses quickly

## 📋 Prerequisites

- **Python 3.7+** with pip
- **Ollama** (for open-source models)
- **Google API Key** (for Gemini access)
- **Tavily API Key** (for AI-optimized search)

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dvirla/ai-model-evaluator.git
   cd ai-model-evaluator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Ollama** (if not already installed)
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull some models
   ollama pull llama3.1:70b
   ollama pull qwen2.5:14b
   ollama pull deepseek-r1:70b
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the interface**
   - Open http://localhost:5001 in your browser

## 🔧 Configuration

Create a `.env` file with the following variables:

```bash
# Required: Google Gemini API key
GOOGLE_API_KEY=your_gemini_api_key_here

# Required: Tavily API key for search functionality
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### Getting API Keys

- **Google Gemini**: Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Tavily Search**: Sign up at [Tavily](https://tavily.com/) for AI-optimized search

## 🎯 Usage Guide

### Step 1: Query Models
1. Enter your prompt/question in the text area
2. Select your preferred model from the dropdown
3. Click "🚀 Query Model" to execute dual-mode comparison
4. Review both responses (with and without search)

### Step 2: Evaluate Responses  
1. Customize the evaluation prompt using template placeholders
2. Select an Ollama model for evaluation
3. Click "🦙 Evaluate with Custom Prompt"
4. Review the evaluation results

### GPU Memory Management
- Click "🔄 Show Loaded Models" to see current GPU usage
- Use "🗑️ Unload" buttons to free specific models
- Click "🧹 Unload All Models" for complete cleanup

### History Management
- Click "📚 Response History" to view past queries  
- Use search to find specific responses
- Click "Load" to restore any previous session

## 📊 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │    │   Flask Backend  │    │   AI Providers  │
│                 │    │                  │    │                 │
│ • Model Select  │◄──►│ • app.py         │◄──►│ • Gemini 2.5    │
│ • Dual Responses│    │ • API Endpoints  │    │ • Ollama Models │
│ • GPU Controls  │    │ • Database       │    │ • Tavily Search │
│ • History View  │    │ • GPU Management │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

- **`app.py`**: Main Flask application with API endpoints
- **`pydantic_ai_wrapper.py`**: Ollama integration with GPU management
- **`templates/new_index.html`**: Responsive web interface
- **SQLite Database**: Stores responses, evaluations, and metadata

## 🔌 API Reference

### POST `/api/query-gemini`
Query any model with dual-mode comparison
```json
{
  "prompt": "Your question here",
  "model_name": "gemini-2.5-pro"
}
```

### POST `/api/evaluate-response`  
Evaluate responses with custom prompts
```json
{
  "response_id": 123,
  "evaluation_prompt": "Rate this response...",
  "ollama_model": "llama3.1:70b"
}
```

### POST `/api/gpu-cleanup`
Unload all models from GPU memory
```json
{
  "success": true,
  "message": "Unloaded 3/3 models",
  "results": [...]
}
```

### GET `/api/loaded-models`
Get currently loaded models and GPU status

### GET `/api/available-models`  
Get all available models (Gemini + Ollama)

## 🎨 Template System

Use these placeholders in evaluation prompts:

- `{original_prompt}` - Insert the user's original question
- `{gemini_without_search_thinking}` - Model's reasoning without search
- `{gemini_without_search_response}` - Final response without search
- `{gemini_with_search_thinking}` - Model's reasoning with search
- `{gemini_with_search_response}` - Final response with search

**Example Evaluation Prompt:**
```
Compare these two responses to: {original_prompt}

Response A (No Search): {gemini_without_search_response}
Response B (With Search): {gemini_with_search_response}

Which response is more accurate and comprehensive? Explain your reasoning.
```

## 🛠️ Development

### Running Tests
```bash
python -m pytest  # or your preferred test command
python test_wrapper.py
```

### Database Schema
- **gemini_responses**: Stores AI model responses with metadata
- **evaluations**: Links evaluation results to response pairs
- **batch_jobs**: Tracks batch processing operations (future feature)

## 🚨 Troubleshooting

### Common Issues

**"Model not found" errors**
- Ensure Ollama models are pulled: `ollama list`
- Check model names match exactly (case-sensitive)

**GPU memory errors**  
- Use GPU cleanup functions before switching models
- Monitor VRAM usage: `nvidia-smi` (NVIDIA GPUs)

**API connection errors**
- Verify API keys in `.env` file
- Check Ollama is running: `ollama serve`

**Search not working**
- Verify Tavily API key is valid
- Check internet connectivity

### Performance Tips

- Use GPU cleanup between different model sizes
- Keep frequently-used models loaded for faster responses
- Use search functionality sparingly for faster responses

## 📈 Roadmap

- [ ] Batch processing for multiple prompts
- [ ] Export results to CSV/JSON formats
- [ ] Advanced filtering and analytics
- [ ] Custom model fine-tuning integration
- [ ] Multi-user support with authentication

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Gemini 2.5 Pro](https://deepmind.google/technologies/gemini/) by Google DeepMind
- [Ollama](https://ollama.ai/) for local model hosting
- [pydantic_ai](https://ai.pydantic.dev/) for AI agent framework
- [Tavily](https://tavily.com/) for AI-optimized search
- [Flask](https://flask.palletsprojects.com/) for the web framework

## 💬 Support

For questions, issues, or contributions:
- 📧 Create an issue on GitHub
- 💭 Join discussions in the repository
- 📖 Check the [documentation](./docs/) for detailed guides

---

**Built with ❤️ for the AI research community**