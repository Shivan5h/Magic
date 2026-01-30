"""Production-grade health monitoring and metrics"""

import time
from typing import Dict, Optional
from datetime import datetime
from pinecone import Pinecone
from groq import Groq
from config import Config

class HealthMonitor:
    """Monitor system health and collect metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.query_count = 0
        self.error_count = 0
        self.total_response_time = 0
        
    def check_all(self) -> Dict:
        """
        Comprehensive health check
        
        Returns:
            Dict with health status of all components
        """
        health = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': int(time.time() - self.start_time),
            'components': {},
            'metrics': self.get_metrics(),
            'overall_status': 'healthy'
        }
        
        # Check Pinecone
        pinecone_health = self._check_pinecone()
        health['components']['pinecone'] = pinecone_health
        
        # Check Groq
        groq_health = self._check_groq()
        health['components']['groq'] = groq_health
        
        # Determine overall status
        if not pinecone_health['healthy'] or not groq_health['healthy']:
            health['overall_status'] = 'unhealthy'
        elif pinecone_health.get('warning') or groq_health.get('warning'):
            health['overall_status'] = 'degraded'
        
        return health
    
    def _check_pinecone(self) -> Dict:
        """Check Pinecone health"""
        try:
            pc = Pinecone(api_key=Config.PINECONE_API_KEY)
            index = pc.Index(Config.PINECONE_INDEX_NAME)
            
            start = time.time()
            stats = index.describe_index_stats()
            latency = time.time() - start
            
            vector_count = stats.get('total_vector_count', 0)
            
            result = {
                'healthy': True,
                'latency_ms': int(latency * 1000),
                'vector_count': vector_count,
                'index_name': Config.PINECONE_INDEX_NAME
            }
            
            # Warnings
            if vector_count == 0:
                result['warning'] = 'No vectors in index'
            elif latency > 1.0:
                result['warning'] = 'High latency'
            
            return result
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def _check_groq(self) -> Dict:
        """Check Groq LLM health"""
        try:
            client = Groq(api_key=Config.GROQ_API_KEY)
            
            start = time.time()
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": "ping"}],
                model=Config.GROQ_MODEL,
                max_tokens=5
            )
            latency = time.time() - start
            
            result = {
                'healthy': True,
                'latency_ms': int(latency * 1000),
                'model': Config.GROQ_MODEL
            }
            
            if latency > 3.0:
                result['warning'] = 'High latency'
            
            return result
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def record_query(self, response_time: float, success: bool):
        """Record query metrics"""
        self.query_count += 1
        if success:
            self.total_response_time += response_time
        else:
            self.error_count += 1
    
    def get_metrics(self) -> Dict:
        """Get application metrics"""
        avg_response_time = (
            self.total_response_time / self.query_count 
            if self.query_count > 0 else 0
        )
        
        error_rate = (
            self.error_count / self.query_count 
            if self.query_count > 0 else 0
        )
        
        return {
            'total_queries': self.query_count,
            'total_errors': self.error_count,
            'error_rate': f"{error_rate * 100:.1f}%",
            'avg_response_time': f"{avg_response_time:.2f}s"
        }


# Global monitor instance
monitor = HealthMonitor()
