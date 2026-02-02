"""
RAG Chatbot for Magicbricks Property Search
Combines RAG pipeline and CLI interface
"""

from typing import List, Dict
from pinecone import Pinecone
from groq import Groq
from config import Config
import sys


# ============================================================================
# RAG PIPELINE
# ============================================================================

class RAGPipeline:
    """RAG pipeline for property search and question answering"""
    
    def __init__(self):
        import logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        try:
            Config.validate()
        except ValueError as e:
            self.logger.error(f"Configuration Error: {e}")
            self.logger.error("Please set up your .env file with required API keys")
            raise
        
        try:
            self.logger.info("Connecting to Pinecone...")
            self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
            self.index = self.pc.Index(Config.PINECONE_INDEX_NAME)
            
            # Verify index exists and get stats
            stats = self.index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            self.logger.info(f"✅ Connected to Pinecone index '{Config.PINECONE_INDEX_NAME}' ({vector_count} vectors)")
            
            if vector_count == 0:
                self.logger.warning("⚠️  Index is empty. Run data_pipeline.py to populate it.")
            
            self.logger.info(f"Using embedding model: {Config.EMBEDDING_MODEL}")
            
            self.logger.info("Initializing Groq LLM client...")
            self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
            
            self.logger.info("✅ RAG Pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG Pipeline: {e}")
            raise
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant property chunks from Pinecone"""
        if top_k is None:
            top_k = Config.TOP_K_RESULTS
        
        try:
            # Use native Pinecone inference API
            query_embeddings = self.pc.inference.embed(
                model=Config.EMBEDDING_MODEL,
                inputs=[query],
                parameters={"input_type": "query"}
            )
            query_embedding = query_embeddings[0].values
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
        except Exception as e:
            print(f"❌ Error in retrieve_relevant_chunks: {e}")
            raise
        
        relevant_chunks = []
        for match in results['matches']:
            chunk = {
                'text': match['metadata'].get('full_text', match['metadata'].get('text', '')),
                'score': match['score'],
                'title': match['metadata'].get('title', ''),
                'location': match['metadata'].get('location', ''),
                'price': match['metadata'].get('price', ''),
                'property_type': match['metadata'].get('property_type', ''),
                'bedrooms': match['metadata'].get('bedrooms', ''),
                'area': match['metadata'].get('area', ''),
                'url': match['metadata'].get('property_url', ''),
            }
            relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def create_context_from_chunks(self, chunks: List[Dict]) -> str:
        """Create formatted context string"""
        if not chunks:
            return "No relevant property information found."
        
        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            context_parts.append(f"Property {idx}:")
            context_parts.append(chunk['text'])
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str, chunks: List[Dict]) -> str:
        """Generate response using Groq LLM with source mapping"""
        system_prompt = """You are a real estate assistant. Answer ONLY using the properties provided in the context below. DO NOT make up information.

CRITICAL RULES:
1. If context has matching properties → List them with details
2. If context has NO matching properties → Say "No properties found with those criteria" and show what IS available
3. ALWAYS use actual prices, locations, and amenities from the context
4. Format each property as ONE bullet point:
   • **[Type]** in [Location] - ₹[Price] | [BHK] | [Area] | [Amenities] [SOURCE:N]"""

        user_prompt = f"""Available Properties in Database:
{context}

User Query: {query}

INSTRUCTIONS:
1. Check if any properties match the user's query
2. List matching properties with ALL details (price, location, BHK, amenities)
3. Add [SOURCE:N] at the end of each property line
4. If NO match found, say so clearly and show what properties ARE available

Answer:"""

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=Config.GROQ_MODEL,
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, user_query: str, top_k: int = None, return_chunks: bool = False) -> Dict:
        """Complete RAG pipeline with production-grade error handling and validation
        
        Args:
            user_query: User's search query
            top_k: Number of chunks to retrieve (default from config)
            return_chunks: Include raw chunks in response
            
        Returns:
            Dict with query results and metadata
        """
        import time
        
        # Input validation
        if not user_query or not isinstance(user_query, str):
            raise ValueError("Query must be a non-empty string")
        
        if len(user_query.strip()) < 3:
            raise ValueError("Query too short. Please provide more details.")
        
        self.logger.info(f"🔍 Query received: '{user_query}'")
        start_time = time.time()
        
        # Retrieve with retry logic
        max_retries = 3
        chunks = None
        
        for retry in range(max_retries):
            try:
                chunks = self.retrieve_relevant_chunks(user_query, top_k)
                break
            except Exception as e:
                if retry == max_retries - 1:
                    self.logger.error(f"Retrieval failed after {max_retries} retries: {e}")
                    return {
                        'query': user_query,
                        'response': "Sorry, I encountered a database error. Please try again.",
                        'error': str(e)
                    }
                self.logger.warning(f"Retry {retry + 1}/{max_retries} for retrieval")
                time.sleep(2 ** retry)
        
        self.logger.info(f"✅ Retrieved {len(chunks)} chunks")
        if chunks:
            self.logger.info(f"📍 Top match: {chunks[0].get('location', 'N/A')} (score: {chunks[0].get('score', 0):.3f})")
        
        # Generate response with retry logic
        for retry in range(max_retries):
            try:
                context = self.create_context_from_chunks(chunks)
                response = self.generate_response(user_query, context, chunks)
                break
            except Exception as e:
                if retry == max_retries - 1:
                    self.logger.error(f"Generation failed after {max_retries} retries: {e}")
                    response = "Sorry, I couldn't generate a response. Please try again."
                self.logger.warning(f"Retry {retry + 1}/{max_retries} for generation")
                time.sleep(2 ** retry)
        
        elapsed = time.time() - start_time
        self.logger.info(f"⏱️ Query completed in {elapsed:.2f}s")
        
        result = {
            'query': user_query,
            'response': response,
            'response_time': f"{elapsed:.2f}s",
            'chunks_retrieved': len(chunks)
        }
        
        if return_chunks:
            result['retrieved_chunks'] = chunks
            result['num_chunks_retrieved'] = len(chunks)
        
        return result
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the Pinecone index"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'index_name': Config.PINECONE_INDEX_NAME,
            }
        except Exception as e:
            return {'error': str(e)}


# ============================================================================
# CLI CHATBOT
# ============================================================================

class ChatbotCLI:
    """Command-line interface for the property chatbot"""
    
    def __init__(self):
        print("🏠 Magicbricks Property Chatbot")
        print("=" * 60)
        print("Initializing chatbot...")
        
        try:
            self.rag = RAGPipeline()
            self.conversation_history = []
            
            stats = self.rag.get_index_stats()
            print(f"✅ Connected to database with {stats.get('total_vectors', 0)} properties")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ Failed to initialize chatbot: {e}")
            print("\nPlease ensure:")
            print("1. You have set up .env file with API keys")
            print("2. You have run: python data_pipeline.py")
            sys.exit(1)
    
    def display_welcome_message(self):
        """Display welcome message"""
        print("\n👋 Welcome! I can help you find properties on Magicbricks.")
        print("\n💡 Example queries:")
        print("   • Show me 3BHK apartments in Bangalore")
        print("   • What properties are available under 1 crore?")
        print("   • Find villas with swimming pool")
        print("   • Properties in Electronic City with gym")
        print("\n📝 Commands:")
        print("   • Type your question and press Enter")
        print("   • 'quit' or 'exit' to end")
        print("   • 'stats' for database statistics")
        print("   • 'clear' to clear history")
        print("=" * 60)
    
    def display_stats(self):
        """Display statistics"""
        stats = self.rag.get_index_stats()
        print("\n📊 Database Statistics:")
        print(f"   Index Name: {stats.get('index_name', 'N/A')}")
        print(f"   Total Properties: {stats.get('total_vectors', 0)}")
        print(f"   Queries in this session: {len(self.conversation_history)}")
        print()
    
    def process_query(self, query: str):
        """Process user query"""
        print("\n🔍 Searching properties...")
        
        try:
            result = self.rag.query(query, return_chunks=False)
            
            self.conversation_history.append({
                'query': query,
                'response': result['response']
            })
            
            print("\n💬 Assistant:")
            print("-" * 60)
            print(result['response'])
            print("-" * 60)
            
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
    
    def run(self):
        """Main chatbot loop"""
        self.display_welcome_message()
        
        while True:
            try:
                user_input = input("\n🙋 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using Magicbricks Chatbot. Goodbye!")
                    break
                
                elif user_input.lower() == 'stats':
                    self.display_stats()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("\n✅ Conversation history cleared")
                    continue
                
                elif user_input.lower() in ['help', 'h']:
                    self.display_welcome_message()
                    continue
                
                self.process_query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chatbot interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_rag_pipeline():
    """Test the RAG pipeline"""
    print("🤖 Testing Magicbricks RAG Pipeline")
    print("=" * 50)
    
    try:
        rag = RAGPipeline()
        
        stats = rag.get_index_stats()
        print(f"\n📊 Index Stats:")
        print(f"   Index: {stats.get('index_name', 'N/A')}")
        print(f"   Total Vectors: {stats.get('total_vectors', 0)}")
        
        test_queries = [
            "Show me 3BHK apartments in Bangalore under 2 crore",
            "What properties are available in Whitefield?",
            "Tell me about luxury villas with swimming pool",
        ]
        
        print("\n" + "=" * 50)
        print("Testing Sample Queries:")
        print("=" * 50)
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 50)
            
            result = rag.query(query, return_chunks=False)
            print(f"💬 Response:\n{result['response']}")
            print()
        
        print("=" * 50)
        print("\n✅ RAG Pipeline test completed successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set up .env file with API keys")
        print("2. Run: python data_pipeline.py")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_rag_pipeline()
    else:
        chatbot = ChatbotCLI()
        chatbot.run()


if __name__ == "__main__":
    main()
