import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Magicbricks RAG Chatbot"""
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'magicbricks-properties')
    
    # Groq Configuration
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = 'llama-3.3-70b-versatile'  # Latest Llama model
    
    # Apify Configuration (Priority)
    APIFY_API_TOKEN = os.getenv('APIFY_API_TOKEN')
    USE_REAL_SCRAPING = os.getenv('USE_REAL_SCRAPING', 'True').lower() == 'true'
    
    # Embedding Configuration
    EMBEDDING_MODEL = 'llama-text-embed-v2'
    EMBEDDING_DIMENSION = 1024  # Llama text embed v2 dimension
    USE_PINECONE_EMBEDDINGS = True  # Use Pinecone Inference API
    
    # Scraping Configuration
    USER_AGENT = os.getenv('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    MAX_PAGES = int(os.getenv('MAX_PAGES', 50))
    REQUEST_DELAY = 2  # seconds between requests
    
    # Data Processing
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    # RAG Configuration
    TOP_K_RESULTS = 5
    TEMPERATURE = 0.1  # Very low for factual responses
    MAX_TOKENS = 1024
    
    # Paths
    DATA_DIR = 'data'
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    
    @classmethod
    def validate(cls):
        """Validate that required API keys are set"""
        required_vars = ['PINECONE_API_KEY', 'GROQ_API_KEY', 'APIFY_API_TOKEN']
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
