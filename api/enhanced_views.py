import openai
import base64
import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from decouple import config
from openai import OpenAI
from .firebase_auth import require_auth
from .document_processor import DocumentProcessor
from .models import UserDocument, UserChatSession, ChatMessage
import logging

logger = logging.getLogger(__name__)

openai.api_key = config('OPENAI_API_KEY')

class EnhancedChatView(APIView):
    """Enhanced chat view with Firebase auth, document processing, and RAG"""
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    
    def __init__(self):
        super().__init__()
        self.document_processor = DocumentProcessor()
    
    @require_auth
    def post(self, request):
        """Handle chat requests with authentication and RAG"""
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
            ##################################################################
            # Accept multiple files
            if medical_report_files:
                for medical_report_file in medical_report_files:
                    document_name = medical_report_file.name
                    document_type = self._determine_document_type(document_name)
                    try:
                        # Process and store document
                        document = self.document_processor.process_document(
                            user_id=user_id,
                            pdf_file=medical_report_file,
                            document_name=document_name,
                            document_type=document_type
                        )
                        # Store system message about document upload
                        ChatMessage.objects.create(
                            session=session,
                            message_type='system',
                            content=f'Document "{document_name}" uploaded and processed successfully.',
                            metadata={'document_id': str(document.id)}
                        )
                        logger.info(f"Document processed for user {user_id}: {document_name}")
                    except Exception as e:
                        logger.error(f"Error processing document: {str(e)}")
                        return Response({
                            'error': f'Failed to process document: {str(e)}'
                        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            ##################################################################
            
            # Prepare user content
            user_content = []
            if user_message:
                user_content.append({"type": "text", "text": user_message})
            
            if image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_base64 = f"data:{image_file.content_type};base64,{image_data}"
                user_content.append({"type": "image_url", "image_url": {"url": image_base64}})
                
                # Add default image prompt if no text provided
                if not user_message:
                    user_content.insert(0, {
                        "type": "text",
                        "text": "Analyze this image of food and describe the food in it and explicitly estimate calories, protein, carbohydrates, fats, and other key nutrients present in the portion shown."
                    })
            
            # Store user message
            ChatMessage.objects.create(
                session=session,
                message_type='user',
                content=user_message or "Image analysis requested",
                metadata={'has_image': bool(image_file)}
            )
            
            # Generate AI response with RAG if applicable
            if user_message and not image_file:
                # Text query - use hybrid search for RAG
                response = self._generate_hybrid_rag_response(user_id, user_message, session)
            elif image_file and user_message:
                # Image + text query - use hybrid search for medical context
                response = self._generate_image_text_hybrid_response(user_id, user_message, image_base64, session)
            elif image_file:
                # Image-only query - use all medical data (current approach)
                medical_chunks = self.document_processor.search_user_documents(
                    user_id=user_id,
                    query="food safety",  # Use a generic query to get relevant medical info
                    top_k=5
                )
                medical_context = self._build_context_from_results(medical_chunks)

                # 2. Build prompt for OpenAI
                prompt = f"""
                The user has the following medical context:
                {medical_context}

                Analyze the uploaded food image and tell if it is safe for the user to eat based on their medical reports. Also, provide one more relevant health tip.
                """

                # 3. Prepare user content for OpenAI
                image_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_base64}}
                ]

                # 4. Get AI response
                try:
                    client = OpenAI(api_key=config('OPENAI_API_KEY'))
                    ai_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful medical nutritionist. Use the user's medical context to analyze food images and provide safety advice."},
                            {"role": "user", "content": image_content}
                        ]
                    )
                    response = ai_response.choices[0].message["content"]
                except Exception as e:
                    logger.error(f"Error generating image+medical response: {str(e)}")
                    response = f"I apologize, but I encountered an error while analyzing the image: {str(e)}"
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
                'session_id': session_id,
                'user_id': user_id
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in enhanced chat: {str(e)}")
            return Response({
                'error': f'Chat error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
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
    
    def _generate_hybrid_rag_response(self, user_id: str, query: str, session: UserChatSession) -> str:
        """Generate response using hybrid search RAG (BM25 + Semantic)"""
        try:
            # Use hybrid search with intelligent weight adjustment
            search_results = self.document_processor._hybrid_search(
                user_id=user_id,
                query=query,
                top_k=3
                # Weights are automatically determined by query analysis
            )
            
            if search_results:
                # Build context from retrieved documents
                context = self._build_context_from_results(search_results)
                
                # Generate response with context
                prompt = f"""Based on the user's medical documents, answer the following question:

Question: {query}

Relevant information from documents:
{context}

Please provide a comprehensive answer based on the information from the user's documents. If the documents don't contain enough information to answer the question, say so clearly."""
                
                client = OpenAI(api_key=config('OPENAI_API_KEY'))
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant. Use the provided document context to answer questions accurately."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                return response.choices[0].message["content"]
            else:
                # Fallback: Try semantic search only if hybrid search failed
                logger.info("Hybrid search returned no results, trying semantic search fallback")
                fallback_results = self.document_processor.search_user_documents(
                    user_id=user_id,
                    query=query,
                    top_k=3
                )
                
                if fallback_results:
                    context = self._build_context_from_results(fallback_results)
                    prompt = f"""Based on the user's medical documents, answer the following question:

Question: {query}

Relevant information from documents:
{context}

Please provide a comprehensive answer based on the information from the user's documents. If the documents don't contain enough information to answer the question, say so clearly."""
                    
                    client = OpenAI(api_key=config('OPENAI_API_KEY'))
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful medical assistant. Use the provided document context to answer questions accurately."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    return response.choices[0].message["content"]
                else:
                    # No relevant documents found - provide general response
                    client = OpenAI(api_key=config('OPENAI_API_KEY'))
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful medical assistant."},
                            {"role": "user", "content": query}
                        ]
                    )
                    
                    return response.choices[0].message["content"]
                
        except Exception as e:
            logger.error(f"Error generating hybrid RAG response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _generate_image_text_hybrid_response(self, user_id: str, query: str, image_base64: str, session: UserChatSession) -> str:
        """Generate response for image + text using hybrid search for medical context"""
        try:
            # Use hybrid search with intelligent weight adjustment for medical context
            medical_chunks = self.document_processor._hybrid_search(
                user_id=user_id,
                query=query,
                top_k=5
                # Weights are automatically determined by query analysis
            )
            
            # Fallback if no medical data found
            if not medical_chunks:
                logger.info("Hybrid search returned no results, trying semantic search fallback")
                medical_chunks = self.document_processor.search_user_documents(
                    user_id=user_id,
                    query=query,
                    top_k=5
                )
            
            medical_context = self._build_context_from_results(medical_chunks)

            # Build prompt for OpenAI
            prompt = f"""
            The user has the following medical context:
            {medical_context}

            User's question: {query}

            Analyze the uploaded image and answer the user's question based on their medical context. Provide relevant health advice considering their medical history.
            """

            # Prepare user content for OpenAI
            image_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]

            # Get AI response
            client = OpenAI(api_key=config('OPENAI_API_KEY'))
            ai_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful medical nutritionist. Use the user's medical context to analyze images and provide personalized health advice."},
                    {"role": "user", "content": image_content}
                ]
            )
            
            return ai_response.choices[0].message["content"]
            
        except Exception as e:
            logger.error(f"Error generating image+text hybrid response: {str(e)}")
            return f"I apologize, but I encountered an error while analyzing the image and processing your question: {str(e)}"
    
    def _generate_rag_response(self, user_id: str, query: str, session: UserChatSession) -> str:
        """Generate response using RAG (Retrieval-Augmented Generation) - Legacy method"""
        try:
            # Search user's documents
            search_results = self.document_processor.search_user_documents(
                user_id=user_id,
                query=query,
                top_k=3
            )
            
            if search_results:
                # Build context from retrieved documents
                context = self._build_context_from_results(search_results)
                
                # Generate response with context
                prompt = f"""Based on the user's medical documents, answer the following question:

Question: {query}

Relevant information from documents:
{context}

Please provide a comprehensive answer based on the information from the user's documents. If the documents don't contain enough information to answer the question, say so clearly."""
                
                client = OpenAI(api_key=config('OPENAI_API_KEY'))
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant. Use the provided document context to answer questions accurately."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                return response.choices[0].message["content"]
            else:
                # No relevant documents found - provide general response
                client = OpenAI(api_key=config('OPENAI_API_KEY'))
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant."},
                        {"role": "user", "content": query}
                    ]
                )
                
                return response.choices[0].message["content"]
                
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _build_context_from_results(self, search_results: list) -> str:
        """Build context string from search results"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            document_name = result['document_name']
            text_content = result['text_content']
            score = result['score']
            
            context_parts.append(f"Document {i}: {document_name} (Relevance: {score:.2f})\n{text_content}\n")
        
        return "\n".join(context_parts)
    
    def _generate_image_response(self, user_content: list, session: UserChatSession) -> str:
        """Generate response for image analysis (without medical context)"""
        try:
            client = OpenAI(api_key=config('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful medical nutritionist. Analyze food images and provide detailed nutritional information."},
                    {"role": "user", "content": user_content}
                ]
            )
            
            return response.choices[0].message["content"]
            
        except Exception as e:
            logger.error(f"Error generating image response: {str(e)}")
            return f"I apologize, but I encountered an error while analyzing the image: {str(e)}"


class DocumentManagementView(APIView):
    """View for managing user documents"""
    
    def __init__(self):
        super().__init__()
        from .document_processor import DocumentProcessor
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
                    'chunk_count': len(doc.vector_ids)
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