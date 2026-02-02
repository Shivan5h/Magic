"""
Complete Data Pipeline for Magicbricks RAG Chatbot
Combines scraping, processing, and embedding generation in one file
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from config import Config

# Apify support (optional)
try:
    from apify_client import ApifyClient
    APIFY_AVAILABLE = True
except ImportError:
    APIFY_AVAILABLE = False
    print("⚠️  apify-client not installed. Real scraping disabled.")


# ============================================================================
# SCRAPER
# ============================================================================

class MagicbricksScraper:
    """Scraper for Magicbricks real estate platform"""
    
    def __init__(self, use_apify: bool = None):
        self.base_url = "https://www.magicbricks.com"
        self.headers = {
            'User-Agent': Config.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        self.properties = []
        
        # Determine if we should use Apify
        if use_apify is None:
            use_apify = Config.USE_REAL_SCRAPING and APIFY_AVAILABLE
        
        self.use_apify = use_apify
        
        # Initialize Apify client if needed
        if self.use_apify:
            if not APIFY_AVAILABLE:
                raise ImportError("apify-client not installed")
            if not Config.APIFY_API_TOKEN:
                raise ValueError("APIFY_API_TOKEN not set in .env")
            self.apify_client = ApifyClient(Config.APIFY_API_TOKEN)
            print("✅ Using Apify for real scraping (Priority mode)")
        else:
            self.apify_client = None
            print("⚠️  Apify not configured - Using sample data fallback")
    
    def scrape_with_apify(self, urls: List[str]) -> List[Dict]:
        """Scrape properties using Apify actor"""
        if not self.apify_client:
            raise RuntimeError("Apify client not initialized")
        
        print(f"🚀 Starting Apify scraping for {len(urls)} URLs...")
        
        run_input = {
            "urls": urls,
            "max_retries_per_url": 2,
            "proxy": {
                "useApifyProxy": True,
                "apifyProxyGroups": ["RESIDENTIAL"],
                "apifyProxyCountry": "IN",
            },
        }
        
        try:
            run = self.apify_client.actor("ecomscrape/magicbricks-property-details-page-scraper").call(
                run_input=run_input
            )
            
            dataset_id = run["defaultDatasetId"]
            print(f"💾 Data: https://console.apify.com/storage/datasets/{dataset_id}")
            
            properties = []
            for item in self.apify_client.dataset(dataset_id).iterate_items():
                property_data = self._transform_apify_data(item)
                properties.append(property_data)
            
            print(f"✅ Scraped {len(properties)} properties via Apify")
            return properties
            
        except Exception as e:
            print(f"❌ Apify error: {e}")
            return []
    
    def _transform_apify_data(self, apify_item: Dict) -> Dict:
        """Transform Apify data to standard format"""
        return {
            'url': apify_item.get('url', ''),
            'scraped_at': datetime.now().isoformat(),
            'title': apify_item.get('title', apify_item.get('propertyName', 'N/A')),
            'price': apify_item.get('price', 'N/A'),
            'location': apify_item.get('location', apify_item.get('locality', 'N/A')),
            'property_type': apify_item.get('propertyType', 'N/A'),
            'bedrooms': apify_item.get('bedrooms', apify_item.get('bhkType', 'N/A')),
            'bathrooms': apify_item.get('bathrooms', 'N/A'),
            'area': apify_item.get('area', apify_item.get('carpetArea', 'N/A')),
            'amenities': apify_item.get('amenities', []),
            'description': apify_item.get('description', apify_item.get('propertyDescription', 'N/A')),
            'builder': apify_item.get('builder', apify_item.get('builderName', 'N/A')),
            'locality_info': apify_item.get('localityInfo', apify_item.get('aboutLocality', 'N/A')),
        }
    
    def get_sample_properties(self) -> List[Dict]:
        """Generate sample property data - 15 diverse Bangalore properties"""
        return [
            # Premium properties
            {
                'url': 'https://www.magicbricks.com/property-sample-1',
                'scraped_at': datetime.now().isoformat(),
                'title': 'Luxury 3BHK Apartment in Whitefield, Bangalore',
                'price': '₹ 1.2 Cr',
                'location': 'Whitefield, Bangalore',
                'property_type': 'Apartment',
                'bedrooms': '3 BHK',
                'bathrooms': '3 Bathrooms',
                'area': '1650 sq.ft',
                'amenities': ['Swimming Pool', 'Gym', 'Club House', 'Power Backup', 'Parking', 'Security', 'Lift'],
                'description': 'Premium 3BHK apartment in Whitefield with modern amenities. Close to IT parks, schools, and shopping centers.',
                'builder': 'Prestige Group',
                'locality_info': 'Whitefield is a major IT hub with excellent infrastructure.',
            },
            {
                'url': 'https://www.magicbricks.com/property-sample-2',
                'scraped_at': datetime.now().isoformat(),
                'title': '2BHK Flat in Electronic City, Bangalore',
                'price': '₹ 75 Lakh',
                'location': 'Electronic City, Bangalore',
                'property_type': 'Apartment',
                'bedrooms': '2 BHK',
                'bathrooms': '2 Bathrooms',
                'area': '1100 sq.ft',
                'amenities': ['Gym', 'Power Backup', 'Parking', 'Security', 'Children Play Area'],
                'description': 'Spacious 2BHK apartment in Electronic City Phase 1. Near major IT companies.',
                'builder': 'Brigade Group',
                'locality_info': "Electronic City is Bangalore's largest IT park.",
            },
            {
                'url': 'https://www.magicbricks.com/property-sample-3',
                'scraped_at': datetime.now().isoformat(),
                'title': '4BHK Villa in Sarjapur Road, Bangalore',
                'price': '₹ 2.5 Cr',
                'location': 'Sarjapur Road, Bangalore',
                'property_type': 'Villa',
                'bedrooms': '4 BHK',
                'bathrooms': '4 Bathrooms',
                'area': '2800 sq.ft',
                'amenities': ['Private Garden', 'Swimming Pool', 'Gym', 'Club House', 'Power Backup', 'Parking', '24x7 Security', 'Gated Community'],
                'description': 'Luxurious 4BHK villa with private garden and premium fittings. Perfect for families.',
                'builder': 'Sobha Developers',
                'locality_info': 'Sarjapur Road is rapidly developing with excellent IT parks and schools.',
            },
            {
                'url': 'https://www.magicbricks.com/property-sample-4',
                'scraped_at': datetime.now().isoformat(),
                'title': '3BHK Apartment in Indiranagar, Bangalore',
                'price': '₹ 2.1 Cr',
                'location': 'Indiranagar, Bangalore',
                'property_type': 'Apartment',
                'bedrooms': '3 BHK',
                'bathrooms': '3 Bathrooms',
                'area': '1800 sq.ft',
                'amenities': ['Gym', 'Club House', 'Power Backup', 'Parking', 'Lift', 'Intercom', 'Piped Gas'],
                'description': 'Premium apartment in the heart of Indiranagar. Walking distance to restaurants and cafes.',
                'builder': 'Shriram Properties',
                'locality_info': 'Indiranagar is one of Bangalore\'s most sought-after neighborhoods.',
            },
            {
                'url': 'https://www.magicbricks.com/property-sample-5',
                'scraped_at': datetime.now().isoformat(),
                'title': '1BHK Studio Apartment in Koramangala, Bangalore',
                'price': '₹ 55 Lakh',
                'location': 'Koramangala, Bangalore',
                'property_type': 'Studio Apartment',
                'bedrooms': '1 BHK',
                'bathrooms': '1 Bathroom',
                'area': '650 sq.ft',
                'amenities': ['Power Backup', 'Parking', 'Security', 'Lift'],
                'description': 'Compact studio apartment perfect for young professionals. Located in the startup hub.',
                'builder': 'Purva Properties',
                'locality_info': 'Koramangala is Bangalore\'s startup district with numerous cafes and restaurants.',
            },
            # Mid-range properties
            {
                'url': 'https://www.magicbricks.com/property-sample-6',
                'scraped_at': datetime.now().isoformat(),
                'title': '2BHK Apartment in HSR Layout, Bangalore',
                'price': '₹ 95 Lakh',
                'location': 'HSR Layout, Bangalore',
                'property_type': 'Apartment',
                'bedrooms': '2 BHK',
                'bathrooms': '2 Bathrooms',
                'area': '1200 sq.ft',
                'amenities': ['Gym', 'Swimming Pool', 'Power Backup', 'Parking', 'Security', 'Lift', 'Club House'],
                'description': 'Well-maintained 2BHK in HSR Layout Sector 2. Great connectivity to ORR and metro.',
                'builder': 'Sobha Developers',
                'locality_info': 'HSR Layout is a well-developed residential area with excellent amenities.',
            },
            {
                'url': 'https://www.magicbricks.com/property-sample-7',
                'scraped_at': datetime.now().isoformat(),
                'title': '3BHK Penthouse in JP Nagar, Bangalore',
                'price': '₹ 1.8 Cr',
                'location': 'JP Nagar, Bangalore',
                'property_type': 'Penthouse',
                'bedrooms': '3 BHK',
                'bathrooms': '3 Bathrooms',
                'area': '2200 sq.ft',
                'amenities': ['Private Terrace', 'Swimming Pool', 'Gym', 'Club House', 'Power Backup', 'Parking', 'Security'],
                'description': 'Stunning penthouse with private terrace and city views. Premium fittings throughout.',
                'builder': 'Puravankara',
                'locality_info': 'JP Nagar is a mature residential area with excellent schools and hospitals.',
            },
            {
                'url': 'https://www.magicbricks.com/property-sample-8',
                'scraped_at': datetime.now().isoformat(),
                'title': '2BHK Flat in Marathahalli, Bangalore',
                'price': '₹ 68 Lakh',
                'location': 'Marathahalli, Bangalore',
                'property_type': 'Apartment',
                'bedrooms': '2 BHK',
                'bathrooms': '2 Bathrooms',
                'area': '950 sq.ft',
                'amenities': ['Power Backup', 'Parking', 'Security', 'Lift', 'Children Play Area'],
                'description': 'Affordable 2BHK near Marathahalli Bridge. Close to offices and tech parks.',
                'builder': 'Salarpuria Sattva',
                'locality_info': 'Marathahalli is a bustling IT hub with great connectivity.',
            },
            # Budget-friendly options
            {
                'url': 'https://www.magicbricks.com/property-sample-9',
                'scraped_at': datetime.now().isoformat(),
                'title': '1BHK Apartment in Bommanahalli, Bangalore',
                'price': '₹ 42 Lakh',
                'location': 'Bommanahalli, Bangalore',
                'property_type': 'Apartment',
                'bedrooms': '1 BHK',
                'bathrooms': '1 Bathroom',
                'area': '580 sq.ft',
                'amenities': ['Power Backup', 'Parking', 'Security'],
                'description': 'Compact 1BHK apartment near Electronic City. Ideal for first-time buyers.',
                'builder': 'Mantri Developers',
                'locality_info': 'Bommanahalli offers affordable housing near Electronic City.',
            },
            {
                'url': 'https://www.magicbricks.com/property-sample-10',
                'scraped_at': datetime.now().isoformat(),
                'title': '2BHK Flat in BTM Layout, Bangalore',
                'price': '₹ 85 Lakh',
                'location': 'BTM Layout, Bangalore',
                'property_type': 'Apartment',
                'bedrooms': '2 BHK',
                'bathrooms': '2 Bathrooms',
                'area': '1050 sq.ft',
                'amenities': ['Gym', 'Power Backup', 'Parking', 'Security', 'Lift'],
                'description': 'Well-located 2BHK in BTM 2nd Stage. Near metro station and shopping centers.',
                'builder': 'Prestige Group',
                'locality_info': 'BTM Layout is a established residential area with good infrastructure.',
            },
            # Luxury segment
            {
                'url': 'https://www.magicbricks.com/property-sample-11',
                'scraped_at': datetime.now().isoformat(),
                'title': '4BHK Luxury Apartment in Bellandur, Bangalore',
                'price': '₹ 3.2 Cr',
                'location': 'Bellandur, Bangalore',
                'property_type': 'Apartment',
                'bedrooms': '4 BHK',
                'bathrooms': '4 Bathrooms',
                'area': '3200 sq.ft',
                'amenities': ['Private Pool', 'Home Theater', 'Gym', 'Club House', 'Concierge Service', 'Power Backup', 'Parking', '24x7 Security', 'Smart Home'],
                'description': 'Ultra-luxury 4BHK with private pool and smart home features. Premium lake-facing views.',
                'builder': 'Embassy Group',
                'locality_info': 'Bellandur is a premium residential area near major IT parks.',
            },
            {
                'url': 'https://www.magicbricks.com/property-sample-12',
                'scraped_at': datetime.now().isoformat(),
                'title': '3BHK Apartment in Hebbal, Bangalore',
                'price': '₹ 1.5 Cr',
                'location': 'Hebbal, Bangalore',
                'property_type': 'Apartment',
                'bedrooms': '3 BHK',
                'bathrooms': '3 Bathrooms',
                'area': '1750 sq.ft',
                'amenities': ['Swimming Pool', 'Gym', 'Club House', 'Power Backup', 'Parking', 'Security', 'Sports Court'],
                'description': 'Spacious 3BHK near Manyata Tech Park. Excellent for IT professionals.',
                'builder': 'Brigade Group',
                'locality_info': 'Hebbal is rapidly developing with excellent connectivity to airport.',
            },
            # More variety
            {
                'url': 'https://www.magicbricks.com/property-sample-13',
                'scraped_at': datetime.now().isoformat(),
                'title': '2BHK Flat in Yelahanka, Bangalore',
                'price': '₹ 62 Lakh',
                'location': 'Yelahanka, Bangalore',
                'property_type': 'Apartment',
                'bedrooms': '2 BHK',
                'bathrooms': '2 Bathrooms',
                'area': '980 sq.ft',
                'amenities': ['Power Backup', 'Parking', 'Security', 'Lift', 'Children Play Area'],
                'description': 'Affordable 2BHK near Yelahanka metro. Close to schools and hospitals.',
                'builder': 'Mahindra Lifespaces',
                'locality_info': 'Yelahanka offers good connectivity to airport and city center.',
            },
            {
                'url': 'https://www.magicbricks.com/property-sample-14',
                'scraped_at': datetime.now().isoformat(),
                'title': '3BHK Villa in Hennur Road, Bangalore',
                'price': '₹ 1.9 Cr',
                'location': 'Hennur Road, Bangalore',
                'property_type': 'Villa',
                'bedrooms': '3 BHK',
                'bathrooms': '3 Bathrooms',
                'area': '2100 sq.ft',
                'amenities': ['Private Garden', 'Swimming Pool', 'Gym', 'Club House', 'Power Backup', 'Parking', 'Gated Community'],
                'description': 'Beautiful villa with private garden. Perfect for families seeking peace.',
                'builder': 'Shriram Properties',
                'locality_info': 'Hennur Road is emerging as a premium residential corridor.',
            },
            {
                'url': 'https://www.magicbricks.com/property-sample-15',
                'scraped_at': datetime.now().isoformat(),
                'title': '2BHK Apartment in Bannerghatta Road, Bangalore',
                'price': '₹ 78 Lakh',
                'location': 'Bannerghatta Road, Bangalore',
                'property_type': 'Apartment',
                'bedrooms': '2 BHK',
                'bathrooms': '2 Bathrooms',
                'area': '1080 sq.ft',
                'amenities': ['Gym', 'Power Backup', 'Parking', 'Security', 'Lift', 'Landscaped Gardens'],
                'description': 'Well-designed 2BHK with modern amenities. Near IIM Bangalore.',
                'builder': 'Sobha Developers',
                'locality_info': 'Bannerghatta Road is a well-connected area with good social infrastructure.',
            }
        ]
    
    def save_to_json(self, filename: str = None):
        """Save scraped data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"magicbricks_properties_{timestamp}.json"
        
        filepath = os.path.join(Config.RAW_DATA_DIR, filename)
        os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.properties, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(self.properties)} properties to {filepath}")
        return filepath


# ============================================================================
# DATA PROCESSOR
# ============================================================================

class DataProcessor:
    """Process and chunk property data"""
    
    def __init__(self):
        self.raw_data = []
        self.processed_chunks = []
    
    def load_raw_data(self, filepath: str = None):
        """Load raw scraped data"""
        if filepath is None:
            files = [f for f in os.listdir(Config.RAW_DATA_DIR) if f.endswith('.json')]
            if not files:
                raise FileNotFoundError("No JSON files found in raw data directory")
            filepath = os.path.join(Config.RAW_DATA_DIR, sorted(files)[-1])
        
        print(f"Loading data from: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        print(f"Loaded {len(self.raw_data)} properties")
        return self.raw_data
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or text == "N/A":
            return ""
        text = ' '.join(text.split())
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text.strip()
    
    def create_property_summary(self, property_data: Dict) -> str:
        """Create comprehensive text summary"""
        parts = []
        
        if property_data.get('title'):
            parts.append(f"Property: {property_data['title']}")
        
        if property_data.get('price') and property_data['price'] != "N/A":
            parts.append(f"Price: {property_data['price']}")
        
        if property_data.get('location') and property_data['location'] != "N/A":
            parts.append(f"Location: {property_data['location']}")
        
        details = []
        if property_data.get('property_type') and property_data['property_type'] != "N/A":
            details.append(f"Type: {property_data['property_type']}")
        if property_data.get('bedrooms') and property_data['bedrooms'] != "N/A":
            details.append(property_data['bedrooms'])
        if property_data.get('bathrooms') and property_data['bathrooms'] != "N/A":
            details.append(property_data['bathrooms'])
        if property_data.get('area') and property_data['area'] != "N/A":
            details.append(f"Area: {property_data['area']}")
        
        if details:
            parts.append("Details: " + ", ".join(details))
        
        if property_data.get('builder') and property_data['builder'] != "N/A":
            parts.append(f"Builder: {property_data['builder']}")
        
        if property_data.get('description') and property_data['description'] != "N/A":
            parts.append(f"Description: {self.clean_text(property_data['description'])}")
        
        if property_data.get('amenities') and property_data['amenities']:
            amenities_text = ", ".join(property_data['amenities'])
            parts.append(f"Amenities: {amenities_text}")
        
        if property_data.get('locality_info') and property_data['locality_info'] != "N/A":
            parts.append(f"Locality: {self.clean_text(property_data['locality_info'])}")
        
        if property_data.get('url'):
            parts.append(f"URL: {property_data['url']}")
        
        return "\n".join(parts)
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks"""
        if chunk_size is None:
            chunk_size = Config.CHUNK_SIZE
        if overlap is None:
            overlap = Config.CHUNK_OVERLAP
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                for punct in ['. ', '.\n', '! ', '?\n']:
                    last_punct = text[start:end].rfind(punct)
                    if last_punct > chunk_size * 0.7:
                        end = start + last_punct + len(punct)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def create_chunks_with_metadata(self, property_data: Dict) -> List[Dict]:
        """Create chunks with metadata"""
        full_text = self.create_property_summary(property_data)
        text_chunks = self.chunk_text(full_text)
        
        chunks_with_metadata = []
        for idx, chunk in enumerate(text_chunks):
            chunk_data = {
                'text': chunk,
                'metadata': {
                    'property_url': property_data.get('url', ''),
                    'title': property_data.get('title', ''),
                    'location': property_data.get('location', ''),
                    'price': property_data.get('price', ''),
                    'property_type': property_data.get('property_type', ''),
                    'bedrooms': property_data.get('bedrooms', ''),
                    'area': property_data.get('area', ''),
                    'chunk_index': idx,
                    'total_chunks': len(text_chunks),
                }
            }
            chunks_with_metadata.append(chunk_data)
        
        return chunks_with_metadata
    
    def process_all_properties(self):
        """Process all properties and create chunks"""
        print(f"Processing {len(self.raw_data)} properties...")
        
        all_chunks = []
        for property_data in self.raw_data:
            chunks = self.create_chunks_with_metadata(property_data)
            all_chunks.extend(chunks)
        
        self.processed_chunks = all_chunks
        print(f"Created {len(all_chunks)} chunks from {len(self.raw_data)} properties")
        return all_chunks
    
    def save_processed_data(self, filename: str = None):
        """Save processed chunks"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_chunks_{timestamp}.json"
        
        filepath = os.path.join(Config.PROCESSED_DATA_DIR, filename)
        os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.processed_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(self.processed_chunks)} chunks to {filepath}")
        return filepath


# ============================================================================
# EMBEDDINGS MANAGER
# ============================================================================

class EmbeddingsManager:
    """Manage embeddings generation and Pinecone operations"""
    
    def __init__(self):
        try:
            Config.validate()
        except ValueError as e:
            print(f"❌ Configuration Error: {e}")
            raise
        
        print(f"Using Pinecone Inference embeddings: {Config.EMBEDDING_MODEL}")
        
        print("Connecting to Pinecone...")
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index_name = Config.PINECONE_INDEX_NAME
        self.index = None
        self.chunks_data = []
    
    def load_processed_chunks(self, filepath: str = None):
        """Load processed chunks"""
        if filepath is None:
            files = [f for f in os.listdir(Config.PROCESSED_DATA_DIR) if f.endswith('.json')]
            if not files:
                raise FileNotFoundError("No JSON files in processed data directory")
            filepath = os.path.join(Config.PROCESSED_DATA_DIR, sorted(files)[-1])
        
        print(f"Loading chunks from: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            self.chunks_data = json.load(f)
        
        print(f"Loaded {len(self.chunks_data)} chunks")
        return self.chunks_data
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings using Pinecone Inference API"""
        print(f"Generating embeddings for {len(texts)} texts using Pinecone Inference...")
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            try:
                # Use native Pinecone inference API
                batch_embeddings = self.pc.inference.embed(
                    model=Config.EMBEDDING_MODEL,
                    inputs=batch,
                    parameters={"input_type": "passage"}
                )
                embeddings.extend([emb.values for emb in batch_embeddings])
            except AttributeError as e:
                print(f"❌ Pinecone inference API not available. Error: {e}")
                print("Your Pinecone client version might not support inference.")
                raise
        
        return embeddings
    
    def create_or_connect_index(self):
        """Create or connect to Pinecone index"""
        print(f"Checking for index: {self.index_name}")
        
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=Config.EMBEDDING_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=Config.PINECONE_ENVIRONMENT or 'us-east-1'
                )
            )
            print(f"✅ Index created")
            time.sleep(5)
        else:
            print(f"✅ Index exists")
        
        self.index = self.pc.Index(self.index_name)
        stats = self.index.describe_index_stats()
        print(f"📊 Vectors: {stats.get('total_vector_count', 0)}")
    
    def prepare_vectors(self, chunks: List[Dict], embeddings: List[List[float]]) -> List[Dict]:
        """Prepare vectors for Pinecone"""
        vectors = []
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector = {
                'id': f"chunk_{idx}_{chunk['metadata'].get('property_url', '').split('/')[-1]}",
                'values': embedding,
                'metadata': {
                    'text': chunk['text'][:1000],
                    'full_text': chunk['text'],
                    'title': chunk['metadata'].get('title', ''),
                    'location': chunk['metadata'].get('location', ''),
                    'price': chunk['metadata'].get('price', ''),
                    'property_type': chunk['metadata'].get('property_type', ''),
                    'bedrooms': chunk['metadata'].get('bedrooms', ''),
                    'area': chunk['metadata'].get('area', ''),
                    'property_url': chunk['metadata'].get('property_url', ''),
                    'chunk_index': chunk['metadata'].get('chunk_index', 0),
                }
            }
            vectors.append(vector)
        
        return vectors
    
    def upload_to_pinecone(self, vectors: List[Dict], batch_size: int = 100):
        """Upload vectors to Pinecone with deduplication"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"📤 Uploading {len(vectors)} vectors with deduplication...")
        
        # Check existing IDs to avoid duplicates
        existing_ids = set()
        try:
            # Fetch sample to check if IDs exist
            for vector in vectors[:min(10, len(vectors))]:
                fetch_result = self.index.fetch(ids=[vector['id']])
                if fetch_result.vectors:
                    existing_ids.add(vector['id'])
        except Exception as e:
            logger.warning(f"Could not check existing vectors: {e}")
        
        # Filter out duplicates
        new_vectors = [v for v in vectors if v['id'] not in existing_ids]
        duplicate_count = len(vectors) - len(new_vectors)
        
        if duplicate_count > 0:
            logger.info(f"⚠️  Skipping {duplicate_count} duplicate vectors")
        
        if not new_vectors:
            logger.info("ℹ️  All vectors already exist in index")
            return
        
        logger.info(f"📤 Uploading {len(new_vectors)} new vectors...")
        
        # Upload with retry logic
        max_retries = 3
        for i in tqdm(range(0, len(new_vectors), batch_size)):
            batch = new_vectors[i:i + batch_size]
            
            for retry in range(max_retries):
                try:
                    self.index.upsert(vectors=batch)
                    break
                except Exception as e:
                    if retry == max_retries - 1:
                        logger.error(f"Failed to upload batch after {max_retries} retries: {e}")
                        raise
                    logger.warning(f"Retry {retry + 1}/{max_retries} for batch upload")
                    time.sleep(2 ** retry)
        
        logger.info("✅ Upload completed")
        time.sleep(2)
        
        stats = self.index.describe_index_stats()
        logger.info(f"📊 Total vectors in index: {stats.get('total_vector_count', 0)}")
    
    def process_and_upload(self):
        """Complete pipeline"""
        self.load_processed_chunks()
        texts = [chunk['text'] for chunk in self.chunks_data]
        embeddings = self.generate_embeddings(texts)
        self.create_or_connect_index()
        vectors = self.prepare_vectors(self.chunks_data, embeddings)
        self.upload_to_pinecone(vectors)
        return vectors


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_complete_pipeline(preferences: Dict = None):
    """Run the complete data pipeline with smart scraping
    
    Args:
        preferences: Optional dict with search criteria
                    {'city': 'Mumbai', 'bhk': '3', 'property_type': 'apartment', 
                     'min_price': 50, 'max_price': 300}
    """
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🏠 Starting Magicbricks Data Pipeline (Production Mode)")
    logger.info("=" * 60)
    
    # Step 1: Smart Scraping
    logger.info("[1/3] SMART SCRAPING")
    logger.info("-" * 60)
    os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
    
    try:
        scraper = MagicbricksScraper(use_apify=True)
    except Exception as e:
        logger.error(f"Failed to initialize scraper: {e}")
        raise RuntimeError("❌ Scraper initialization failed. Check Apify configuration.")
    
    if not scraper.use_apify:
        raise RuntimeError("❌ Production mode requires Apify. Set APIFY_API_TOKEN in .env")
    
    # Try Apify first, fallback to sample data if trial expired
    logger.info("📋 Attempting to scrape real properties via Apify...")
    
    # These are real Bangalore property URLs for Apify
    bangalore_urls = [
        "https://www.magicbricks.com/propertyDetails/3-BHK-Apartment-FOR-Sale-Whitefield-in-Bangalore",
        "https://www.magicbricks.com/propertyDetails/2-BHK-Flat-FOR-Sale-Koramangala-in-Bangalore",
        "https://www.magicbricks.com/propertyDetails/3-BHK-Villa-FOR-Sale-Sarjapur-Road-in-Bangalore",
        "https://www.magicbricks.com/propertyDetails/2-BHK-Apartment-FOR-Sale-Electronic-City-in-Bangalore",
    ]
    
    try:
        logger.info(f"🚀 Trying to scrape {len(bangalore_urls)} real properties...")
        properties = scraper.scrape_with_apify(bangalore_urls)
        
        if not properties:
            raise RuntimeError("No properties returned from Apify")
        
        logger.info(f"✅ Scraped {len(properties)} real properties via Apify!")
        
    except Exception as e:
        logger.warning(f"⚠️  Apify scraping failed: {e}")
        logger.info("📦 Using sample data fallback for demo...")
        properties = scraper.get_sample_properties()
        logger.info(f"✅ Using {len(properties)} sample Bangalore properties")
    
    if not properties:
        raise RuntimeError("❌ No properties available (Apify failed and no sample data).")
    
    logger.info(f"✅ Total properties scraped: {len(properties)}")
    scraper.properties = properties
    scraper.save_to_json()
    
    # Step 2: Processing
    print("\n[2/3] PROCESSING")
    print("-" * 60)
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    
    processor = DataProcessor()
    processor.load_raw_data()
    processor.process_all_properties()
    processor.save_processed_data()
    
    # Step 3: Embeddings
    print("\n[3/3] EMBEDDINGS")
    print("-" * 60)
    
    try:
        manager = EmbeddingsManager()
        manager.process_and_upload()
        
        print("\n" + "=" * 60)
        print("✅ Pipeline completed successfully!")
        print("=" * 60)
        print("\n💡 Next steps:")
        print("   • python rag_chatbot.py (CLI)")

        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set up .env file with API keys")
        print("2. Valid Pinecone and Groq credentials")


if __name__ == "__main__":
    run_complete_pipeline()
