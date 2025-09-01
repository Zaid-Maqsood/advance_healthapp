import PyPDF2
import re
import uuid
from typing import List, Dict, Any
from .models import UserDocument, DocumentChunk
from .pinecone_utils import get_pinecone_manager
import openai
from decouple import config
import logging

logger = logging.getLogger(__name__)

openai.api_key = config('OPENAI_API_KEY')

class DocumentProcessor:
    """Process documents and store them in vector database"""
    
    def __init__(self):
        self.pinecone_manager = get_pinecone_manager()
        self.chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def create_text_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_endings = ['.', '!', '?', '\n\n']
                for ending in sentence_endings:
                    last_ending = text.rfind(ending, start, end)
                    if last_ending > start + self.chunk_size * 0.7:  # Only break if we're not too early
                        end = last_ending + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks using OpenAI"""
        embeddings = []
        
        for text in texts:
            try:
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embedding = response['data'][0]['embedding']
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                raise
        
        return embeddings
    
    def store_document_vectors(self, user_id: str, document_name: str, 
                             chunks: List[str], embeddings: List[List[float]], 
                             document_type: str = 'other') -> UserDocument:
        """Store document chunks in Pinecone and create database records"""
        
        # Create document record
        document = UserDocument.objects.create(
            user_id=user_id,
            document_name=document_name,
            document_type=document_type,
            extracted_text="\n\n".join(chunks),
            processing_status='processing'
        )
        
        vector_ids = []
        chunk_records = []
        
        try:
            # Prepare vectors for Pinecone
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"doc_{document.id}_chunk_{i}_user_{user_id}"
                
                vector_data = {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "user_id": user_id,
                        "document_id": str(document.id),
                        "document_name": document_name,
                        "document_type": document_type,
                        "chunk_index": i,
                        "text_content": chunk,
                        "source": "user_document"
                    }
                }
                vectors.append(vector_data)
                
                # Create chunk record
                chunk_record = DocumentChunk(
                    document=document,
                    chunk_index=i,
                    text_content=chunk,
                    vector_id=vector_id
                )
                chunk_records.append(chunk_record)
                vector_ids.append(vector_id)
            
            # Store vectors in Pinecone
            success = self.pinecone_manager.upsert_vectors(vectors)
            
            if success:
                # Save chunk records to database
                DocumentChunk.objects.bulk_create(chunk_records)
                
                # Update document with vector IDs
                document.vector_ids = vector_ids
                document.processing_status = 'completed'
                document.save()
                
                logger.info(f"Successfully stored document {document.id} with {len(chunks)} chunks")
                return document
            else:
                document.processing_status = 'failed'
                document.save()
                raise Exception("Failed to store vectors in Pinecone")
                
        except Exception as e:
            document.processing_status = 'failed'
            document.save()
            logger.error(f"Error storing document vectors: {str(e)}")
            raise
    
    def process_document(self, user_id: str, pdf_file, document_name: str, 
                        document_type: str = 'other') -> UserDocument:
        """Complete document processing pipeline"""
        
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_file)
            
            # Create text chunks
            chunks = self.create_text_chunks(text)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # Store in vector database
            document = self.store_document_vectors(
                user_id=user_id,
                document_name=document_name,
                chunks=chunks,
                embeddings=embeddings,
                document_type=document_type
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
    
    def search_user_documents(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search user's documents for relevant information"""
        """full code"""
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            # Search in Pinecone with user filter
            results = self.pinecone_manager.query_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"user_id": user_id}
            )
            
            # Format results
            formatted_results = []
            for match in results:
                formatted_results.append({
                    'id': match.id,
                    'score': match.score,
                    'text_content': match.metadata.get('text_content', ''),
                    'document_name': match.metadata.get('document_name', ''),
                    'document_type': match.metadata.get('document_type', ''),
                    'chunk_index': match.metadata.get('chunk_index', 0),
                    'metadata': match.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching user documents: {str(e)}")
            return []
    
    def get_user_documents(self, user_id: str) -> List[UserDocument]:
        """Get all documents for a user"""
        return UserDocument.objects.filter(
            user_id=user_id,
            processing_status='completed'
        ).order_by('-upload_date')
    
    def delete_user_document(self, user_id: str, document_id: str) -> bool:
        """Delete a user's document and its vectors"""
        try:
            document = UserDocument.objects.get(id=document_id, user_id=user_id)
            
            # Delete vectors from Pinecone
            if document.vector_ids:
                self.pinecone_manager.delete_vectors(document.vector_ids)
            
            # Delete from database
            document.delete()
            
            logger.info(f"Successfully deleted document {document_id} for user {user_id}")
            return True
            
        except UserDocument.DoesNotExist:
            logger.warning(f"Document {document_id} not found for user {user_id}")
            return False
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
