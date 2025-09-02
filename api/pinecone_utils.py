import os
import pinecone
from django.conf import settings
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PineconeManager:
    def __init__(self):
        self.api_key = settings.PINECONE_API_KEY
        self.environment = settings.PINECONE_ENVIRONMENT
        self.index_name = settings.PINECONE_INDEX_NAME
        self.pc = None
        self.index = None
        self._initialize_pinecone()
        # Ensure index is created and accessible
        self.create_index_if_not_exists()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client using the new API"""
        try:
            self.pc = pinecone.Pinecone(api_key=self.api_key)
            logger.info(f"Successfully initialized Pinecone client")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def create_index_if_not_exists(self):
        """Create Pinecone index for OpenAI embeddings if it doesn't exist"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                # Create index for OpenAI embeddings (ada-002, 1536 dims)
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
                    
                
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            else:
                logger.info(f"Index {self.index_name} already exists")
            # Get the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Successfully connected to index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise

    
    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """Upsert vectors to Pinecone index"""
        try:
            if not vectors:
                logger.warning("No vectors provided for upsert")
                return False
            
            # Ensure index is available
            if self.index is None:
                logger.error("Index is not initialized")
                self.create_index_if_not_exists()
                
            self.index.upsert(vectors=vectors)
            logger.info(f"Successfully upserted {len(vectors)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            return False
    
    def query_vectors(self, query_vector: List[float], top_k: int = 5, 
                     include_metadata: bool = True, filter: Optional[Dict] = None) -> List[Dict]:
        """Query vectors from Pinecone index"""
        try:
            # Ensure index is available
            if self.index is None:
                logger.error("Index is not initialized")
                self.create_index_if_not_exists()
                
            query_response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                filter=filter
            )
            
            return query_response.matches
            
        except Exception as e:
            logger.error(f"Error querying vectors: {str(e)}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Pinecone index"""
        try:
            if not ids:
                logger.warning("No IDs provided for deletion")
                return False
            
            # Ensure index is available
            if self.index is None:
                logger.error("Index is not initialized")
                self.create_index_if_not_exists()
                
            self.index.delete(ids=ids)
            logger.info(f"Successfully deleted {len(ids)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index"""
        try:
            # Ensure index is available
            if self.index is None:
                logger.error("Index is not initialized")
                self.create_index_if_not_exists()
                
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}
    
    def delete_index(self) -> bool:
        """Delete the entire Pinecone index"""
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"Successfully deleted index: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            return False

# Global instance
pinecone_manager = None

def get_pinecone_manager() -> PineconeManager:
    """Get or create a global Pinecone manager instance"""
    global pinecone_manager
    if pinecone_manager is None:
        pinecone_manager = PineconeManager()
    return pinecone_manager
