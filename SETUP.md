# AI Model Response Evaluator - Complete Setup Guide

This guide provides step-by-step instructions for setting up the AI Model Response Evaluator on different platforms.

## 📋 System Requirements

### Minimum Requirements
- **Python 3.7+** (Python 3.8+ recommended)
- **4GB RAM** (8GB+ recommended for better performance)
- **2GB free disk space**
- **Internet connection** (for API access and model downloads)

### Recommended for GPU Features
- **NVIDIA GPU** with 8GB+ VRAM (for large models)
- **CUDA-compatible GPU** for optimal Ollama performance
- **16GB+ system RAM** for handling multiple models

### Supported Platforms
- ✅ Linux (Ubuntu 18.04+, CentOS 7+, etc.)
- ✅ macOS (10.14+)  
- ✅ Windows 10/11
- ✅ Docker (all platforms)

## 🐍 Python Environment Setup

### Option 1: System Python (Simple)
```bash
# Check Python version
python3 --version  # Should be 3.7+

# Install pip if not available
python3 -m ensurepip --upgrade

# Clone and setup
git clone https://github.com/dvirla/ai-model-evaluator.git
cd ai-model-evaluator
pip3 install -r requirements.txt
```

### Option 2: Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv ai-evaluator-env

# Activate environment
# On Linux/Mac:
source ai-evaluator-env/bin/activate
# On Windows:
ai-evaluator-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Conda Environment
```bash
# Create conda environment
conda create -n ai-evaluator python=3.9
conda activate ai-evaluator

# Install dependencies
pip install -r requirements.txt
```

## 🦙 Ollama Installation & Configuration

### Linux Installation
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Test installation
ollama --version
```

### macOS Installation
```bash
# Download from website or use homebrew
brew install ollama

# Or download from: https://ollama.ai/download
# Start Ollama (will auto-start on system boot)
```

### Windows Installation
```bash
# Download installer from: https://ollama.ai/download
# Run the installer and follow instructions
# Ollama will start automatically
```

### Pull Recommended Models
```bash
# Essential models (choose based on your hardware)
ollama pull llama3.1:8b      # Smaller, faster (4GB VRAM)
ollama pull llama3.1:70b     # Better quality (40GB VRAM)
ollama pull qwen2.5:14b      # Good balance (10GB VRAM)
ollama pull deepseek-r1:70b  # Latest reasoning model (40GB VRAM)

# Lightweight options for limited hardware
ollama pull gemma2:2b        # Very fast, minimal VRAM
ollama pull phi3:mini        # Microsoft's compact model

# Verify models are installed
ollama list
```

## 🔑 API Keys Setup

### 1. Google Gemini API Key

1. **Visit Google AI Studio**
   - Go to https://aistudio.google.com/app/apikey
   - Sign in with your Google account

2. **Create API Key**
   - Click "Create API Key"
   - Select your Google Cloud project (or create one)
   - Copy the generated API key

3. **Verify Access**
   ```bash
   # Test API key (replace with your key)
   curl -H "Content-Type: application/json" \
        -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
        -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=YOUR_API_KEY"
   ```

### 2. Tavily Search API Key

1. **Sign up for Tavily**
   - Go to https://tavily.com/
   - Create an account
   - Choose the free tier (1000 searches/month)

2. **Get API Key**
   - Navigate to your dashboard
   - Copy your API key from the API section

3. **Test API Key**
   ```bash
   # Test Tavily API (replace with your key)
   curl -X POST "https://api.tavily.com/search" \
        -H "Content-Type: application/json" \
        -d '{"api_key": "YOUR_TAVILY_API_KEY", "query": "test", "max_results": 1}'
   ```

## ⚙️ Environment Configuration

### Create .env File
```bash
# Copy example file
cp .env.example .env

# Edit with your preferred editor
nano .env
# OR
vim .env
# OR
code .env
```

### Complete .env Configuration
```bash
# Required: Google Gemini API access
GOOGLE_API_KEY=your_actual_gemini_api_key_here

# Required: Tavily search API
TAVILY_API_KEY=your_actual_tavily_api_key_here

# Optional: Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_KEEP_ALIVE=5m

# Optional: Application settings
FLASK_ENV=development
FLASK_DEBUG=True
DATABASE_PATH=./new_query_history.db

# Optional: Logging configuration
LOG_LEVEL=INFO
LOG_FILE=app.log
```

## 🚀 Running the Application

### Development Mode
```bash
# Ensure environment is activated
source ai-evaluator-env/bin/activate  # Linux/Mac
# OR
ai-evaluator-env\Scripts\activate     # Windows

# Start the application
python app.py

# Application will be available at:
# http://localhost:5001
```

### Production Mode (Linux/Mac)
```bash
# Using gunicorn for production
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:5001 app:app

# With systemd service (optional)
sudo cp deployment/ai-evaluator.service /etc/systemd/system/
sudo systemctl enable ai-evaluator
sudo systemctl start ai-evaluator
```

### Docker Deployment
```bash
# Build image
docker build -t ai-model-evaluator .

# Run container
docker run -d \
  --name ai-evaluator \
  -p 5001:5001 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  ai-model-evaluator

# With GPU support (NVIDIA Docker)
docker run --gpus all -d \
  --name ai-evaluator \
  -p 5001:5001 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  ai-model-evaluator
```

## 🔧 Hardware-Specific Setup

### NVIDIA GPU Configuration
```bash
# Install CUDA (if not already installed)
# Ubuntu/Debian:
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify CUDA installation
nvidia-smi
nvcc --version

# Test GPU with Ollama
ollama pull llama3.1:8b
ollama run llama3.1:8b "test gpu"
```

### AMD GPU Configuration
```bash
# Install ROCm (for AMD GPUs)
# Ubuntu:
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dev

# Configure Ollama for AMD
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

### Apple Silicon (M1/M2) Configuration
```bash
# Ollama will automatically use Metal performance shaders
# No additional configuration needed

# Verify Metal acceleration
system_profiler SPDisplaysDataType | grep Metal

# Models will automatically use GPU acceleration
ollama pull llama3.1:8b
```

## 🧪 Testing Your Setup

### 1. Basic Functionality Test
```bash
# Run application tests
python -m pytest  # or your preferred test command
python test_wrapper.py

# Should see output like:
# ✅ Database schema created successfully
# ✅ API endpoints working
# ✅ GPU management functional
```

### 2. Integration Test
```bash
# Start the application
python app.py

# Open browser to http://localhost:5001
# Try a simple query:
# 1. Enter "What is artificial intelligence?"
# 2. Select "gemini-2.5-pro" 
# 3. Click "Query Model"
# 4. Verify both responses appear
```

### 3. GPU Test (if applicable)
```bash
# Check GPU utilization
nvidia-smi  # For NVIDIA
rocm-smi    # For AMD

# In the web interface:
# 1. Go to GPU Memory Management section
# 2. Click "Show Loaded Models"
# 3. Verify models appear
# 4. Test "Unload All Models"
```

## 🚨 Troubleshooting

### Common Installation Issues

**Python Version Errors**
```bash
# If python3 not found
sudo apt install python3 python3-pip  # Ubuntu/Debian
brew install python3                   # macOS
```

**Permission Errors**
```bash
# Fix pip permissions
pip install --user -r requirements.txt

# Or use sudo (not recommended)
sudo pip install -r requirements.txt
```

**Ollama Connection Errors**
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama manually
ollama serve

# Check port binding
netstat -tlnp | grep 11434
```

**API Key Issues**
```bash
# Verify .env file is loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('GOOGLE_API_KEY:', bool(os.getenv('GOOGLE_API_KEY')))"

# Should print: GOOGLE_API_KEY: True
```

### Performance Optimization

**Memory Usage**
```bash
# Monitor memory usage
htop
# Or
free -h

# Reduce memory usage
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30s
```

**Disk Space**
```bash
# Check Ollama model storage
du -sh ~/.ollama/

# Clean up unused models
ollama rm model-name
```

**Network Optimization**
```bash
# For slow API responses, adjust timeouts
export REQUEST_TIMEOUT=60
export API_RETRY_DELAY=5
```

## 📊 Monitoring & Logging

### Application Logs
```bash
# View real-time logs
tail -f app.log

# Check for errors
grep ERROR app.log

# Monitor API calls
grep "API" app.log
```

### System Monitoring
```bash
# GPU monitoring
watch -n 1 nvidia-smi

# CPU/Memory monitoring
htop

# Network monitoring
iftop
```

## 🔄 Updates & Maintenance

### Updating the Application
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Update Ollama models
ollama pull llama3.1:70b

# Restart application
```

### Database Maintenance
```bash
# Backup database
cp new_query_history.db new_query_history.db.backup

# Check database size
ls -lh new_query_history.db

# Compact database (if needed)
sqlite3 new_query_history.db "VACUUM;"
```

---

## 🆘 Getting Help

If you encounter issues not covered in this guide:

1. **Check the logs**: Look for error messages in the application logs
2. **Verify requirements**: Ensure all dependencies are correctly installed
3. **Test components**: Run individual tests to isolate issues
4. **Create an issue**: Submit detailed bug reports on GitHub
5. **Join discussions**: Participate in community discussions

For additional support, visit our [GitHub Issues](https://github.com/dvirla/ai-model-evaluator/issues) page.