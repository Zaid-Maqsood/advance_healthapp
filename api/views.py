import openai
import base64
import PyPDF2
import re
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from uuid import uuid4
from decouple import config
from .pinecone_utils import get_pinecone_manager


openai.api_key = config('OPENAI_API_KEY')

# Temporary in-memory session store (for dev/testing only)
session_history = {}

# Medical data storage per session
session_medical_data = {}

# Maximum number of sessions to keep in memory
MAX_SESSIONS = 20

class MedicalReportProcessor:
    """Simple medical report processor"""
    
    def extract_medical_data(self, pdf_file) -> dict:
        """Extract basic medical information from PDF"""
        try:
            # Read PDF content
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Simple extraction of key medical info
            medical_data = {
                "conditions": self.extract_conditions(text),
                "medications": self.extract_medications(text),
                "allergies": self.extract_allergies(text),
                "raw_text": text[:500]  # Store first 500 chars
            }
            
            return medical_data
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return {"error": f"Failed to process PDF: {str(e)}"}
    
    def extract_conditions(self, text: str) -> list:
        """Extract medical conditions"""
        conditions = []
        text_lower = text.lower()
        
        # Common conditions to look for
        common_conditions = [
            'diabetes', 'hypertension', 'heart disease', 'kidney disease', 
            'asthma', 'allergies', 'celiac disease', 'high cholesterol'
        ]
        
        for condition in common_conditions:
            if condition in text_lower:
                conditions.append(condition)
        
        return conditions
    
    def extract_medications(self, text: str) -> list:
        """Extract medications"""
        medications = []
        text_lower = text.lower()
        
        # Common medications
        common_meds = [
            'metformin', 'insulin', 'lisinopril', 'aspirin', 'warfarin'
        ]
        
        for med in common_meds:
            if med in text_lower:
                medications.append(med)
        
        return medications
    
    def extract_allergies(self, text: str) -> list:
        """Extract allergies"""
        allergies = []
        text_lower = text.lower()
        
        # Common allergies
        common_allergies = [
            'peanuts', 'tree nuts', 'dairy', 'eggs', 'soy', 'wheat', 'fish', 'shellfish'
        ]
        
        for allergy in common_allergies:
            if allergy in text_lower:
                allergies.append(allergy)
        
        return allergies

class ChatView(APIView):
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def __init__(self):
        super().__init__()
        self.medical_processor = MedicalReportProcessor()

    def post(self, request):
        # Clear old sessions if we have too many
        if len(session_history) > MAX_SESSIONS:
            print(f"Clearing old sessions. Current count: {len(session_history)}")
            # Keep only the 10 most recent sessions
            recent_sessions = dict(list(session_history.items())[-10:])
            recent_medical_data = {k: session_medical_data.get(k) for k in recent_sessions.keys()}
            session_history.clear()
            session_medical_data.clear()
            session_history.update(recent_sessions)
            session_medical_data.update(recent_medical_data)
            print(f"Sessions after cleanup: {len(session_history)}")
        
        session_id = request.data.get('session_id') or str(uuid4())
        user_message = request.data.get('message', '').strip()
        image_file = request.FILES.get('image', None)
        medical_report_file = request.FILES.get('medical_report', None)

        # Process medical report if uploaded
        if medical_report_file:
            print(f"Processing medical report for session: {session_id}")
            medical_data = self.medical_processor.extract_medical_data(medical_report_file)
            session_medical_data[session_id] = medical_data
            print(f"Medical data extracted: {medical_data}")

        # Get medical context for this session
        medical_context = session_medical_data.get(session_id, {})

        # Initialize chat history if not present
        if session_id not in session_history:
            session_history[session_id] = [{
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            }]

        try:
            user_content = []

            if user_message:
                user_content.append({"type": "text", "text": user_message})

            if image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_base64 = f"data:{image_file.content_type};base64,{image_data}"
                user_content.append({"type": "image_url", "image_url": {"url": image_base64}})

                # If no text provided, add default image prompt
                if not user_message:
                    user_content.insert(0, {
                        "type": "text",
                        "text": "Analyse this image of food and describe the food in it and explicitly estimate calories, protein, carbohydrates, fats, and other key nutrients present in the portion shown."
                    })

            # Append user message
            session_history[session_id].append({
                "role": "user",
                "content": user_content
            })

            # Debug
            print(f"\nSession ID: {session_id}")
            print(f"Total sessions in memory: {len(session_history)}")
            print(f"Messages in current session: {len(session_history[session_id])}")
            print(f"Medical context available: {bool(medical_context)}")

            # Call OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=session_history[session_id]
            )

            assistant_message = response.choices[0].message
            session_history[session_id].append({
                "role": "assistant",
                "content": assistant_message["content"]
            })

            # Convert assistant response
            reply = ""
            if isinstance(assistant_message["content"], list):
                for part in assistant_message["content"]:
                    if part["type"] == "text":
                        reply += part["text"] + "\n"
            else:
                reply = assistant_message["content"]

            # Generate personalized advice if medical context is available
            if medical_context and not medical_context.get("error"):
                # Create personalized prompt
                medical_info = self.create_medical_summary(medical_context)
                personalized_prompt = f"""Based on the user's medical information: {medical_info}

Analyze the food and provide:
1. Is it safe for this person to eat? (Yes/No with brief reason)
2. Complete nutritional breakdown (calories, protein, carbs, fat, etc.)
3. Any specific warnings based on their medical conditions

Keep the response clear and concise."""

                # Get personalized response
                personalized_response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a medical nutritionist."},
                        {"role": "user", "content": f"Food analysis: {reply}\n\n{personalized_prompt}"}
                    ]
                )
                
                final_response = personalized_response.choices[0].message["content"]
                medical_used = True
            else:
                final_response = reply
                medical_used = False

            if len(final_response) > 2000:
                final_response = final_response[:2000] + "..."

        except openai.error.InvalidRequestError as e:
            # Handle specific OpenAI API errors like "message too long"
            if "message too long" in str(e).lower():
                # Clear history and start fresh
                session_history[session_id] = [{
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                }]
                final_response = "The conversation was too long. I've started a new session. Please try your question again."
                medical_used = False
            else:
                final_response = f"OpenAI API Error: {str(e)}"
                medical_used = False
        except Exception as e:
            final_response = f"Error: {str(e)}"
            medical_used = False

        return Response({
            'response': final_response.strip(),
            'session_id': session_id,
            'medical_context_used': medical_used,
            'medical_data_available': bool(medical_context and not medical_context.get("error"))
        }, status=status.HTTP_200_OK)

    def create_medical_summary(self, medical_context: dict) -> str:
        """Create a simple summary of medical information"""
        conditions = medical_context.get("conditions", [])
        medications = medical_context.get("medications", [])
        allergies = medical_context.get("allergies", [])
        
        summary_parts = []
        
        if conditions:
            summary_parts.append(f"Medical conditions: {', '.join(conditions)}")
        if medications:
            summary_parts.append(f"Medications: {', '.join(medications)}")
        if allergies:
            summary_parts.append(f"Allergies: {', '.join(allergies)}")
        
        return "; ".join(summary_parts) if summary_parts else "No specific medical information available"


class VectorSearchView(APIView):
    """View for vector search operations using Pinecone"""
    
    def post(self, request):
        """Store vectors and perform similarity search"""
        try:
            action = request.data.get('action')
            
            if action == 'store':
                return self._store_vectors(request)
            elif action == 'search':
                return self._search_vectors(request)
            elif action == 'stats':
                return self._get_stats(request)
            else:
                return Response({
                    'error': 'Invalid action. Use "store", "search", or "stats"'
                }, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            return Response({
                'error': f'Vector search error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _store_vectors(self, request):
        """Store vectors in Pinecone"""
        vectors_data = request.data.get('vectors', [])
        
        if not vectors_data:
            return Response({
                'error': 'No vectors provided'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            manager = get_pinecone_manager()
            
            # Ensure index exists
            manager.create_index_if_not_exists(dimension=1536, metric="cosine")
            
            # Store vectors
            success = manager.upsert_vectors(vectors_data)
            
            if success:
                return Response({
                    'message': f'Successfully stored {len(vectors_data)} vectors',
                    'vectors_stored': len(vectors_data)
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'error': 'Failed to store vectors'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            return Response({
                'error': f'Error storing vectors: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _search_vectors(self, request):
        """Search for similar vectors"""
        query_vector = request.data.get('query_vector')
        top_k = request.data.get('top_k', 5)
        filter_dict = request.data.get('filter', None)
        
        if not query_vector:
            return Response({
                'error': 'No query vector provided'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            manager = get_pinecone_manager()
            
            # Perform search
            results = manager.query_vectors(
                query_vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = []
            for match in results:
                formatted_results.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                })
            
            return Response({
                'results': formatted_results,
                'total_matches': len(formatted_results)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'error': f'Error searching vectors: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _get_stats(self, request):
        """Get Pinecone index statistics"""
        try:
            manager = get_pinecone_manager()
            stats = manager.get_index_stats()
            
            return Response({
                'stats': stats
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'error': f'Error getting stats: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class VectorDemoView(APIView):
    """Demo view to show how to use Pinecone with OpenAI embeddings"""
    
    def post(self, request):
        """Demo: Store text embeddings and search"""
        try:
            texts = request.data.get('texts', [])
            query_text = request.data.get('query_text', '')
            
            if not texts:
                return Response({
                    'error': 'No texts provided for embedding'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Generate embeddings using OpenAI (still using text-embedding-ada-002 for compatibility)
            embeddings = []
            for i, text in enumerate(texts):
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embedding = response['data'][0]['embedding']
                
                embeddings.append({
                    'id': f'text-{i}',
                    'values': embedding,
                    'metadata': {
                        'text': text,
                        'source': 'demo'
                    }
                })
            
            # Store in Pinecone
            manager = get_pinecone_manager()
            manager.create_index_if_not_exists()
            manager.upsert_vectors(embeddings)
            
            # Search if query provided
            search_results = []
            if query_text:
                # Generate query embedding
                query_response = openai.Embedding.create(
                    input=query_text,
                    model="text-embedding-ada-002"
                )
                query_embedding = query_response['data'][0]['embedding']
                
                # Search
                results = manager.query_vectors(query_embedding, top_k=3)
                search_results = [
                    {
                        'id': match.id,
                        'score': match.score,
                        'text': match.metadata.get('text', ''),
                        'metadata': match.metadata
                    }
                    for match in results
                ]
            
            return Response({
                'message': f'Stored {len(texts)} text embeddings',
                'texts_stored': texts,
                'search_results': search_results if search_results else None
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'error': f'Demo error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

####################################################################
from openai import OpenAI
import base64
import uuid
from datetime import datetime, timedelta
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from decouple import config
from .firebase_auth import require_auth
from .document_processor import DocumentProcessor
from .models import UserDocument, UserChatSession, ChatMessage
import logging
from .pinecone_utils import get_pinecone_manager

logger = logging.getLogger(__name__)

client = OpenAI(api_key=config('OPENAI_API_KEY'))

class EnhancedChatView(APIView):
    """Enhanced chat view with medical context awareness and hybrid search optimization"""
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    
    def __init__(self):
        super().__init__()
        self.document_processor = DocumentProcessor()
        self.pinecone_manager = get_pinecone_manager()
    
    @require_auth
    def post(self, request):
        """Handle chat requests with medical context awareness"""
        try:
            user_id = request.user_id
            user_message = request.data.get('message', '').strip()
            image_file = request.FILES.get('image', None)
            medical_report_files = request.FILES.getlist('medical_report', None)
            session_id = request.data.get('session_id') or str(uuid.uuid4())
            
            # Get or create chat session
            session, created = UserChatSession.objects.get_or_create(
                session_id=session_id,
                user_id=user_id,
                defaults={'is_active': True}
            )
            
            # Update session activity
            session.last_activity = datetime.now()
            session.save()
            
            # Clean up old sessions and check inactivity
            self._cleanup_old_sessions(user_id)
            if self._end_session_if_inactive(session):
                session = UserChatSession.objects.create(
                    session_id=str(uuid.uuid4()),
                    user_id=user_id,
                    is_active=True
                )
            
            # Process medical report uploads
            if medical_report_files:
                self._process_medical_reports(medical_report_files, user_id, session)
            
            # Store user message
            ChatMessage.objects.create(
                session=session,
                message_type='user',
                content=user_message or "Image analysis requested",
                metadata={'has_image': bool(image_file)}
            )
            
            # Generate AI response based on content type and medical context
            if user_message and not image_file:
                response = self._generate_text_response(user_id, user_message, session)
            elif image_file:
                response = self._generate_image_response(image_file, user_id, user_message, session)
            else:
                response = "Please provide a message or image to analyze."
            
            # Store assistant response
            ChatMessage.objects.create(
                session=session,
                message_type='assistant',
                content=response
            )
            
            return Response({
                'response': response,
                'session_id': session.session_id,
                'user_id': user_id
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in enhanced chat: {str(e)}")
            return Response({
                'error': f'Chat error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @require_auth
    def delete(self, request):
        """End a chat session and clear its history"""
        try:
            user_id = request.user_id
            session_id = request.data.get('session_id')
            
            if not session_id:
                return Response({
                    'error': 'Session ID is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            success = self._end_chat_session(session_id, user_id)
            
            if success:
                return Response({
                    'message': 'Chat session ended and history cleared successfully',
                    'session_id': session_id,
                    'user_id': user_id
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'error': 'Failed to end chat session'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Error ending chat session: {str(e)}")
            return Response({
                'error': f'Session error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _process_medical_reports(self, medical_report_files, user_id: str, session: UserChatSession):
        """Process uploaded medical reports"""
        for medical_report_file in medical_report_files:
            document_name = medical_report_file.name
            document_type = self._determine_document_type(document_name)
            
            try:
                document = self.document_processor.process_document(
                    user_id=user_id,
                    pdf_file=medical_report_file,
                    document_name=document_name,
                    document_type=document_type
                )
                
                ChatMessage.objects.create(
                    session=session,
                    message_type='system',
                    content=f'Document "{document_name}" uploaded and processed successfully.',
                    metadata={'document_id': str(document.id)}
                )
                
                logger.info(f"Document processed for user {user_id}: {document_name}")
                
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                raise e
    
    def _generate_text_response(self, user_id: str, query: str, session: UserChatSession) -> str:
        """Generate text response with medical context awareness"""
        try:
            # Check if user has medical documents
            user_documents = self.document_processor.get_user_documents(user_id)
            
            if user_documents:
                # User has medical context - use RAG with hybrid search
                return self._generate_medical_context_response(user_id, query, session)
            else:
                # No medical context - respond as nutritionist
                return self._generate_nutritionist_response(query)
                
        except Exception as e:
            logger.error(f"Error generating text response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _generate_image_response(self, image_file, user_id: str, user_message: str, session: UserChatSession) -> str:
        """Generate image response with medical context awareness"""
        try:
            # Check if user has medical documents
            user_documents = self.document_processor.get_user_documents(user_id)
            
            logger.info(f"Image analysis for user {user_id}: Found {len(user_documents)} medical documents")
            for doc in user_documents:
                logger.info(f"Document: {doc.document_name}, Status: {doc.processing_status}, Chunks: {len(doc.vector_ids) if doc.vector_ids else 0}")
            
            if user_documents:
                # User has medical context - provide personalized food safety advice
                logger.info(f"Using medical context path for user {user_id}")
                return self._generate_medical_image_response(image_file, user_id, user_message, session)
            else:
                # No medical context - provide general nutritional analysis
                logger.info(f"Using general analysis path for user {user_id} - no medical documents found")
                return self._generate_general_image_response(image_file, user_message, session)
                
        except Exception as e:
            logger.error(f"Error generating image response: {str(e)}")
            return f"I apologize, but I encountered an error while analyzing the image: {str(e)}"
    
    def _generate_medical_context_response(self, user_id: str, query: str, session: UserChatSession) -> str:
        """Generate response using medical context with hybrid search optimization"""
        try:
            # Get recent session context (last 20 messages)
            recent_context = self._get_recent_session_context(session, max_messages=20)
            
            # Perform hybrid search with optimized parameters
            search_results = self._perform_optimized_hybrid_search(user_id, query)
            
            if search_results:
                # Build context from retrieved documents
                context = self._build_context_from_results(search_results)
                
                # Generate response with medical context
                prompt = f"""Based on the user's medical documents and recent chat context, answer the following question:

Question: {query}

Relevant medical information:
{context}

Recent chat context (last 20 messages):
{self._format_chat_context(recent_context)}

Please provide a comprehensive answer as a medical nutritionist, considering the user's medical context and recent conversation. If the documents don't contain enough information, say so clearly."""
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful medical nutritionist. Use the provided medical context to answer questions accurately and provide personalized health advice."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                return response.choices[0].message.content
            else:
                # No relevant documents found - provide general nutritionist response
                return self._generate_nutritionist_response(query)
                
        except Exception as e:
            logger.error(f"Error generating medical context response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _generate_medical_image_response(self, image_file, user_id: str, user_message: str, session: UserChatSession) -> str:
        """Generate personalized food safety response using medical context"""
        try:
            # Get recent session context
            recent_context = self._get_recent_session_context(session, max_messages=20)
            
            # Perform hybrid search for medical context
            query = user_message if user_message else "food safety medical conditions allergies"
            medical_chunks = self._perform_optimized_hybrid_search(user_id, query)
            
            # Build medical context string
            medical_context = self._build_context_from_results(medical_chunks)
            
            # Prepare image data
            image_file.seek(0)
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
            image_base64 = f"data:{image_file.content_type};base64,{image_data}"
            
            logger.info(f"Processing medical image: {image_file.name}, content_type: {image_file.content_type}, size: {len(image_data)}")
            
            # Build comprehensive prompt
            prompt = f"""Based on the user's medical reports and recent chat context, analyze this food image and provide:

1. FOOD SAFETY ASSESSMENT: Is this food safe for the user to eat based on their medical conditions? Answer with YES or NO first.
2. NUTRITIONAL ANALYSIS: Detailed breakdown of calories, protein, carbs, fats, vitamins, and minerals
3. PERSONALIZED HEALTH ADVICE: One specific health tip based on the food and user's medical context

User's Medical Context:
{medical_context}

Recent Chat Context:
{self._format_chat_context(recent_context)}

Please analyze the image and provide a comprehensive, personalized response starting with the safety assessment."""

            # Get AI response
            image_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]
            
            ai_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful medical nutritionist. Analyze food images and provide personalized safety advice based on medical context."},
                    {"role": "user", "content": image_content}
                ]
            )
            
            return ai_response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating medical image response: {str(e)}")
            return f"I apologize, but I encountered an error while analyzing the image: {str(e)}"
    
    def _generate_general_image_response(self, image_file, user_message: str, session: UserChatSession) -> str:
        """Generate general nutritional response without medical context"""
        try:
            # Prepare image data
            image_file.seek(0)
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
            image_base64 = f"data:{image_file.content_type};base64,{image_data}"
            
            logger.info(f"Processing general image: {image_file.name}, content_type: {image_file.content_type}, size: {len(image_data)}")
            
            # Build prompt for general analysis
            prompt = "Analyze this food image and provide: 1. Food identification and description, 2. Detailed nutritional information including calories, protein, carbohydrates, fats, vitamins, and minerals."
            
            image_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]
            
            # Get AI response
            ai_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful nutritionist. Analyze food images and provide detailed nutritional information."},
                    {"role": "user", "content": image_content}
                ]
            )
            
            return ai_response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating general image response: {str(e)}")
            return f"I apologize, but I encountered an error while analyzing the image: {str(e)}"
    
    def _generate_nutritionist_response(self, query: str) -> str:
        """Generate general nutritionist response without medical context"""
        try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                    {"role": "system", "content": "You are a helpful nutritionist. Answer questions about nutrition, health, and wellness in a friendly, conversational way."},
                        {"role": "user", "content": query}
                    ]
                )
                
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error generating nutritionist response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _perform_optimized_hybrid_search(self, user_id: str, query: str, top_k: int = 5) -> list:
        """Perform hybrid search with automatically optimized parameters"""
        try:
            # Start with balanced hybrid search
            alpha = 0.5
            results = self.document_processor.search_user_documents(
                user_id=user_id,
                query=query,
                top_k=top_k,
                alpha=alpha
            )
            
            # If results are insufficient, try semantic-only search
            if len(results) < 2:
                results = self.document_processor.search_user_documents_semantic_only(
                    user_id=user_id,
                    query=query,
                    top_k=top_k
                )
            
            # If still insufficient, try keyword-only search
            if len(results) < 2:
                results = self.document_processor.search_user_documents_keyword_only(
                    user_id=user_id,
                    query=query,
                    top_k=top_k
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in optimized hybrid search: {str(e)}")
            return []
    
    def _get_recent_session_context(self, session: UserChatSession, max_messages: int = 20) -> list:
        """Get recent messages from current session (last 20 messages)"""
        try:
            recent_messages = ChatMessage.objects.filter(
                session=session
            ).exclude(
                message_type='system'
            ).order_by('-timestamp')[:max_messages]
            
            context_messages = []
            for msg in reversed(recent_messages):
                if msg.message_type == 'user':
                    context_messages.append({"role": "user", "content": msg.content})
                elif msg.message_type == 'assistant':
                    context_messages.append({"role": "assistant", "content": msg.content})
            
            return context_messages
            
        except Exception as e:
            logger.error(f"Error getting recent session context: {str(e)}")
            return []
    
    def _format_chat_context(self, context_messages: list) -> str:
        """Format chat context for AI prompt"""
        if not context_messages:
            return "No recent conversation context."
        
        formatted_context = []
        for msg in context_messages[-5:]:  # Last 5 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_context.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted_context)
    
    def _build_context_from_results(self, search_results: list) -> str:
        """Build context string from search results"""
        if not search_results:
            return "No relevant medical documents found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            document_name = result['document_name']
            text_content = result['text_content']
            score = result['score']
            
            context_parts.append(f"Document {i}: {document_name} (Relevance: {score:.2f})\n{text_content}\n")
        
        return "\n".join(context_parts)
    
    def _determine_document_type(self, filename: str) -> str:
        """Determine document type based on filename"""
        filename_lower = filename.lower()
        
        if any(keyword in filename_lower for keyword in ['blood', 'lab', 'test', 'result']):
            return 'lab_result'
        elif any(keyword in filename_lower for keyword in ['prescription', 'medication', 'rx']):
            return 'prescription'
        elif any(keyword in filename_lower for keyword in ['xray', 'mri', 'ct', 'imaging', 'scan']):
            return 'imaging'
        elif any(keyword in filename_lower for keyword in ['report', 'medical', 'health']):
            return 'medical_report'
        else:
            return 'other'
    
    def _cleanup_old_sessions(self, user_id: str, max_sessions: int = 3):
        """Keep only recent sessions and clear old ones"""
        try:
            all_sessions = UserChatSession.objects.filter(
                user_id=user_id
            ).order_by('-last_activity')
            
            if all_sessions.count() > max_sessions:
                sessions_to_delete = all_sessions[max_sessions:]
                for session in sessions_to_delete:
                    ChatMessage.objects.filter(session=session).delete()
                    session.delete()
                    
                logger.info(f"Cleaned up {len(sessions_to_delete)} old sessions for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {str(e)}")
    
    def _end_session_if_inactive(self, session: UserChatSession, max_idle_minutes: int = 30):
        """End session if it's been inactive for too long"""
        try:
            idle_time = datetime.now() - session.last_activity.replace(tzinfo=None)
            if idle_time > timedelta(minutes=max_idle_minutes):
                ChatMessage.objects.filter(session=session).delete()
                session.is_active = False
                session.save()
                
                logger.info(f"Session {session.session_id} ended due to inactivity")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking session inactivity: {str(e)}")
            return False
    
    def _clear_session_history(self, session_id: str, user_id: str) -> bool:
        """Clear all messages from a specific session"""
        try:
            session = UserChatSession.objects.get(
                session_id=session_id,
                user_id=user_id
            )
            
            deleted_count = ChatMessage.objects.filter(session=session).delete()[0]
            session.is_active = False
            session.save()
            
            logger.info(f"Session {session_id} cleared for user {user_id}. Deleted {deleted_count} messages.")
            return True
            
        except UserChatSession.DoesNotExist:
            logger.warning(f"Session {session_id} not found for user {user_id}")
            return False
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {str(e)}")
            return False
    
    def _end_chat_session(self, session_id: str, user_id: str) -> bool:
        """End a chat session and clear its history"""
        try:
            success = self._clear_session_history(session_id, user_id)
            
            if success:
                logger.info(f"Chat session {session_id} ended and cleared for user {user_id}")
                return True
            else:
                logger.warning(f"Failed to clear session {session_id} for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error ending chat session {session_id}: {str(e)}")
            return False


class DocumentManagementView(APIView):
    """View for managing user documents"""
    
    def __init__(self):
        super().__init__()
        self.document_processor = DocumentProcessor()
    
    @require_auth
    def get(self, request):
        """Get user's documents"""
        try:
            user_id = request.user_id
            documents = self.document_processor.get_user_documents(user_id)
            
            document_list = []
            for doc in documents:
                document_list.append({
                    'id': str(doc.id),
                    'document_name': doc.document_name,
                    'document_type': doc.document_type,
                    'upload_date': doc.upload_date.isoformat(),
                    'processing_status': doc.processing_status,
                    'chunk_count': len(doc.vector_ids) if doc.vector_ids else 0
                })
            
            return Response({
                'documents': document_list,
                'total_count': len(document_list)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting user documents: {str(e)}")
            return Response({
                'error': f'Error retrieving documents: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @require_auth
    def delete(self, request, document_id):
        """Delete a user's document"""
        try:
            user_id = request.user_id
            
            success = self.document_processor.delete_user_document(user_id, document_id)
            
            if success:
                return Response({
                    'message': 'Document deleted successfully'
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'error': 'Document not found or could not be deleted'
                }, status=status.HTTP_404_NOT_FOUND)
                
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return Response({
                'error': f'Error deleting document: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatHistoryView(APIView):
    """View for retrieving chat history"""
    
    @require_auth
    def get(self, request):
        """Get user's chat history"""
        try:
            user_id = request.user_id
            session_id = request.GET.get('session_id')
            
            if session_id:
                # Get specific session
                try:
                    session = UserChatSession.objects.get(
                        session_id=session_id,
                        user_id=user_id
                    )
                    messages = session.messages.all().order_by('timestamp')
                except UserChatSession.DoesNotExist:
                    return Response({
                        'error': 'Session not found'
                    }, status=status.HTTP_404_NOT_FOUND)
            else:
                # Get all user sessions
                sessions = UserChatSession.objects.filter(
                    user_id=user_id,
                    is_active=True
                ).order_by('-last_activity')
                
                session_list = []
                for session in sessions:
                    session_list.append({
                        'session_id': session.session_id,
                        'created_at': session.created_at.isoformat(),
                        'last_activity': session.last_activity.isoformat(),
                        'message_count': session.messages.count()
                    })
                
                return Response({
                    'sessions': session_list
                }, status=status.HTTP_200_OK)
            
            # Format messages
            message_list = []
            for message in messages:
                message_list.append({
                    'id': str(message.id),
                    'type': message.message_type,
                    'content': message.content,
                    'timestamp': message.timestamp.isoformat(),
                    'metadata': message.metadata
                })
            
            return Response({
                'session_id': session_id,
                'messages': message_list
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return Response({
                'error': f'Error retrieving chat history: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class HybridSearchView(APIView):
    """View for hybrid search operations with automatic parameter optimization"""
    
    def __init__(self):
        super().__init__()
        self.document_processor = DocumentProcessor()
        self.pinecone_manager = get_pinecone_manager()
    
    def post(self, request):
        """Perform hybrid search with automatic parameter optimization"""
        try:
            query = request.data.get('query', '').strip()
            user_id = request.data.get('user_id')
            top_k = request.data.get('top_k', 5)
            
            if not query:
                return Response({
                    'error': 'Query is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if not user_id:
                return Response({
                    'error': 'User ID is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Perform optimized hybrid search
            results = self._perform_optimized_hybrid_search(user_id, query, top_k)
            
            return Response({
                'search_type': 'Optimized Hybrid Search',
                'query': query,
                'user_id': user_id,
                'results': results,
                'total_matches': len(results),
                'search_parameters': {
                    'top_k': top_k,
                    'optimization': 'Automatic parameter selection based on results'
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return Response({
                'error': f'Search error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _perform_optimized_hybrid_search(self, user_id: str, query: str, top_k: int = 5) -> list:
        """Perform hybrid search with automatically optimized parameters"""
        try:
            # Start with balanced hybrid search
            alpha = 0.5
            results = self.document_processor.search_user_documents(
                user_id=user_id,
                query=query,
                top_k=top_k,
                alpha=alpha
            )
            
            # If results are insufficient, try semantic-only search
            if len(results) < 2:
                results = self.document_processor.search_user_documents_semantic_only(
                    user_id=user_id,
                    query=query,
                    top_k=top_k
                )
            
            # If still insufficient, try keyword-only search
            if len(results) < 2:
                results = self.document_processor.search_user_documents_keyword_only(
                    user_id=user_id,
                    query=query,
                    top_k=top_k
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in optimized hybrid search: {str(e)}")
            return []
