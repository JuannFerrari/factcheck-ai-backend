# FactCheck AI Backend

[![CI Pipeline](https://github.com/JuannFerrari/factcheck-ai-backend/workflows/CI%20Pipeline/badge.svg)](https://github.com/JuannFerrari/factcheck-ai-backend/actions)

**Frontend Repository**: [factcheck-ai-frontend](https://github.com/JuannFerrari/factcheck-ai-frontend)

An AI-powered fact-checking API that uses Retrieval-Augmented Generation (RAG) with web search capabilities, vector database storage, and Hugging Face Inference Providers to verify claims against reliable sources.

## 🚀 Features

- **AI-Powered Fact Checking**: Uses Meta Llama 3 8B Instruct model via Hugging Face for intelligent claim verification
- **Web Search Integration**: Leverages Serper.dev API to search for relevant sources and evidence
- **Vector Database Storage**: Stores fact-check results with embeddings for similarity search and caching
- **Hybrid Search**: Combines vector similarity search with web search for enhanced retrieval
- **RAG Architecture**: Combines retrieval (web search + vector search) with generation (AI analysis) for comprehensive fact-checking
- **Rate Limiting**: Built-in rate limiting with configurable limits per minute and per second
- **API Key Authentication**: Secure API key-based authentication
- **Comprehensive Logging**: Structured logging with configurable log levels
- **CORS Support**: Configurable CORS origins for frontend integration
- **Testing**: Comprehensive test suite with unit, integration, and end-to-end tests

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │───▶│  Rate Limiter   │───▶│  Hybrid Search  │
│                 │    │  (slowapi)      │    │  Service        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  AI Model       │◀───│  Sources        │◀───│  Vector DB      │
│  (Llama 3)      │    │  Processing     │    │  + Web Search   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐     ┌─────────────────┐    ┌─────────────────┐
│  Hugging Face   │───▶│  Fact-Check     │    │  Neon PostgreSQL│
│  Inference API  │     │  Response       │    │  + pgvector     │
└─────────────────┘     └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- Hugging Face API key
- Serper.dev API key
- Neon PostgreSQL database (with pgvector extension)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd factcheck-ai-backend
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` with your API keys and configuration:
   ```env
   # Required API Keys
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   
   # Database Configuration
   NEON_DATABASE_URL=postgresql://username:password@host:port/database
   
   # Optional: Custom API key for authentication
   API_KEY=your_custom_api_key_here
   ```

5. **Initialize the database**
   ```bash
   python setup_neon_db.py
   ```

## 🚀 Running the Application

### Development Mode
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Authentication

All API endpoints require authentication using an API key in the request header:
```
X-API-Key: your_api_key_here
```

### Endpoints

#### POST `/api/v1/factcheck`

Fact-check a claim using AI, web search, and vector database.

**Request Body:**
```json
{
  "claim": "The Earth is flat"
}
```

**Response:**
```json
{
  "verdict": "False",
  "confidence": 95,
  "reasoning": "The claim that the Earth is flat is false. Multiple scientific sources confirm that the Earth is an oblate spheroid...",
  "tldr": "The Earth is not flat. Scientific evidence confirms it's an oblate spheroid.",
  "sources": [
    {
      "title": "NASA - Earth",
      "url": "https://www.nasa.gov/earth",
      "snippet": "Earth is the third planet from the Sun and the only astronomical object known to harbor life..."
    }
  ],
  "claim": "The Earth is flat"
}
```

**Rate Limits:**
- Fact-check endpoint: 10 requests per minute, 2 requests per second
- Vector database endpoints: 30 requests per minute, 5 requests per second

**Response Codes:**
- `200`: Success
- `401`: Invalid or missing API key
- `429`: Rate limit exceeded
- `500`: Internal server error

## 🧪 Testing

### Run all tests
```bash
python run_tests.py
```

### Run specific test types
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# End-to-end tests only
pytest tests/e2e/ -v

# With coverage report
pytest --cov=app --cov-report=html
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HUGGINGFACE_API_KEY` | Hugging Face API key | - | Yes |
| `HUGGINGFACE_BASE_URL` | Hugging Face base URL | `https://router.huggingface.co/v1` | No |
| `HUGGINGFACE_MODEL` | AI model to use | `meta-llama/Meta-Llama-3-8B-Instruct` | No |
| `SERPER_API_KEY` | Serper.dev API key | - | Yes |
| `NEON_DATABASE_URL` | Neon PostgreSQL connection string | - | Yes |
| `API_KEY` | Custom API key for authentication | - | No |

| `VECTOR_SIMILARITY_THRESHOLD` | Vector similarity threshold | `0.55` | No |
| `MAX_VECTOR_RESULTS` | Max vector search results | `3` | No |
| `ENABLE_VECTOR_SEARCH` | Enable vector search | `true` | No |
| `ENABLE_VECTOR_STORAGE` | Enable vector storage | `true` | No |
| `RATE_LIMIT_PER_MINUTE` | Rate limit per minute | `60` | No |
| `CORS_ORIGINS` | Allowed CORS origins | `["http://localhost:3000"]` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the [tests README](tests/README.md) for testing documentation
- Review the configuration options in `env.example`
