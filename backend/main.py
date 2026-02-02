"""
FastAPI Backend for Magicbricks RAG Chatbot (Production-Ready)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import os
import time
import logging
from pathlib import Path

from rag_chatbot import RAGPipeline
from config import Config
from health_monitor import monitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Magicbricks RAG Chatbot API",
    description="Production-grade property search with RAG",
    version="1.0.0"
)

# CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Initialize RAG pipeline (lazy loading)
rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline():
    """Lazy load RAG pipeline with error handling"""
    global rag_pipeline
    if rag_pipeline is None:
        try:
            logger.info("Initializing RAG pipeline...")
            rag_pipeline = RAGPipeline()
            logger.info("✅ RAG pipeline ready")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"System initialization failed: {str(e)}"
            )
    return rag_pipeline


# Request/Response models with validation
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    
    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or just whitespace')
        return v.strip()


class QueryResponse(BaseModel):
    success: bool
    response: str
    sources: list = []
    response_time: Optional[str] = None
    chunks_retrieved: Optional[int] = None
    error: Optional[str] = None


# Routes
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Magicbricks RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "query": "/query (POST)"
        }
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check with component status"""
    try:
        health_data = monitor.check_all()
        
        status_code = 200
        if health_data['overall_status'] == 'unhealthy':
            status_code = 503
        elif health_data['overall_status'] == 'degraded':
            status_code = 200  # Still serving but with warnings
        
        return JSONResponse(
            status_code=status_code,
            content=health_data
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/query", response_model=QueryResponse)
async def query_properties(request: QueryRequest):
    """
    Process user query and return response with sources (production-ready)
    """
    start_time = time.time()
    success = False
    
    try:
        # Input validation (Pydantic already validates)
        logger.info(f"📥 Query received: '{request.query}'")
        
        # Get RAG pipeline
        rag = get_rag_pipeline()
        
        # Process query with monitoring
        result = rag.query(request.query, top_k=request.top_k, return_chunks=True)
        
        # Format sources
        sources = []
        for i, chunk in enumerate(result.get('retrieved_chunks', []), 1):
            sources.append({
                "rank": i,
                "score": float(chunk.get('score', 0)),
                "text": chunk.get('text', ''),
                "metadata": chunk.get('metadata', {})
            })
        
        # Record success metrics
        elapsed = time.time() - start_time
        success = True
        monitor.record_query(elapsed, success)
        
        logger.info(f"✅ Query completed in {elapsed:.2f}s")
        
        return QueryResponse(
            success=True,
            response=result['response'],
            sources=sources,
            response_time=f"{elapsed:.2f}s",
            chunks_retrieved=len(sources)
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        # Validation errors
        logger.warning(f"Validation error: {e}")
        elapsed = time.time() - start_time
        monitor.record_query(elapsed, False)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # System errors
        logger.error(f"Query failed: {e}", exc_info=True)
        elapsed = time.time() - start_time
        monitor.record_query(elapsed, False)
        
        return QueryResponse(
            success=False,
            response="Sorry, I encountered an error processing your request. Please try again.",
            error=str(e)
        )


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        # Return stats without initializing RAG pipeline
        stats = {
            "status": "ready",
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.GROQ_MODEL,
            "index_name": Config.PINECONE_INDEX_NAME
        }
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Magicbricks RAG Chatbot Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

