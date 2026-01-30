# 🏠 Magicbricks RAG Chatbot (Production-Ready)

A production-grade Retrieval-Augmented Generation (RAG) chatbot for property search using:
- **Pinecone** for vector storage
- **Groq** for LLM generation  
- **Apify** for web scraping
- **FastAPI** for REST API
- **Smart scraping** for dynamic property discovery

## 🎯 Key Features

### Production-Grade Architecture
- ✅ Smart scraper with dynamic search
- ✅ Comprehensive error handling with retry logic
- ✅ Structured logging throughout
- ✅ Health monitoring & metrics
- ✅ Input validation (Pydantic)
- ✅ Deduplication in vector DB
- ✅ Configuration validation
- ✅ FastAPI with CORS support

### User Features
- Real-time property search
- Natural language queries
- Source attribution
- Multi-city support
- Price/BHK/location filtering

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:
```env
# Required
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key

# For scraping (optional but recommended)
APIFY_API_TOKEN=your_apify_token
```

### 3. Validate Setup

```bash
python config_validator.py
```

### 4. Populate Database

```bash
# Default: Scrapes Mumbai, Bangalore, Delhi
python data_pipeline.py

```

### 5. Start Server

```bash
# Development
uvicorn main:app --reload --port 8000

# Production  
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 6. Open Browser
Navigate to: `http://localhost:8000`

## 📁 Project Structure

```
Magic bricks/
├── main.py                    # FastAPI app (production-ready)
├── rag_chatbot.py            # RAG pipeline with retry logic
├── data_pipeline.py          # ETL with smart scraper
├── smart_scraper.py          # Dynamic property search
├── config.py                 # Configuration
├── config_validator.py       # Validation tool
├── health_monitor.py         # Health checks & metrics  
├── check_database.py         # DB inspection tool
├── PRODUCTION_CHECKLIST.md   # Deployment guide
└── static/                   # Frontend UI
```

## 🛠️ Usage Examples

### Check System Health
```bash
curl http://localhost:8000/health
```

### Query Properties
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "3BHK in Mumbai under 2 crores", "top_k": 5}'
```

### Custom Property Search
```python
from smart_scraper import SmartPropertySearch

searcher = SmartPropertySearch()
properties = searcher.scrape_user_preferences({
    'city': 'Mumbai',
    'bhk': '3',
    'property_type': 'apartment',
    'min_price': 50,
    'max_price': 200
})
```

## 🏗️ Production Architecture

```
User Query → FastAPI → Input Validation → RAG Pipeline
                                            ↓
Query Embedding → Pinecone Search → LLM Generation → Response
    (retry)         (with dedup)        (retry)
```

### Data Flow
1. **Smart Scraper**: User preferences → Search URL → Extract listings → Scrape details
2. **Processing**: Raw HTML → Cleaned data → Chunks → Embeddings
3. **Storage**: Deduplicated vectors → Pinecone (with metadata)
4. **Retrieval**: Query embedding → Top-K search → Context building
5. **Generation**: Context + Query → Groq LLM → Response

## 🔍 Monitoring & Debugging

### Check Database
```bash
python check_database.py
```

### Validate Config
```bash
python config_validator.py
```

### Health Check
```bash
curl http://localhost:8000/health | jq
```

## 📊 Performance

- **Query Latency**: ~2-3s (retrieval + generation)
- **Pinecone Query**: ~150ms
- **Groq Generation**: ~1-2s
- **Throughput**: ~20 queries/min (single instance)

## 🚨 Common Issues

| Issue | Solution |
|-------|----------|
| "No properties found" | Run `python data_pipeline.py` |
| "Apify credits exhausted" | Add credits at console.apify.com |
| "Pinecone connection failed" | Verify API key in `.env` |
| Slow responses | Check `/health` for component latency |

## 🔒 Security (Production)

Before deploying:
- [ ] Restrict CORS origins (not `*`)
- [ ] Add rate limiting
- [ ] Enable API authentication
- [ ] Use secrets manager (not `.env`)
- [ ] Enable HTTPS

See [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) for full list.

## 📈 Scaling

### Horizontal Scaling
```bash
uvicorn main:app --workers 4 --host 0.0.0.0
```

### Caching Layer
- Add Redis for frequent queries
- TTL: 1 hour recommended

### Database
- Pinecone serverless auto-scales
- Consider dedicated for >1M vectors

## ⚙️ Configuration

Environment variables (`.env`):
```env
# Required
PINECONE_API_KEY=
GROQ_API_KEY=

# Optional
APIFY_API_TOKEN=
PINECONE_INDEX_NAME=magic
GROQ_MODEL=llama-3.3-70b-versatile
TEMPERATURE=0.1
MAX_TOKENS=1000
TOP_K=5
```

## 🔒 Privacy & Legal

This project includes sample data for demonstration. For real scraping:
- Review the [APIFY_GUIDE.md](APIFY_GUIDE.md) for instructions
- Ensure compliance with Magicbricks' terms of service
- Check robots.txt before scraping
- Use Apify for reliable, legal scraping

## Thank YOU
