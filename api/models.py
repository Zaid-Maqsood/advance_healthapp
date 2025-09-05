from django.db import models
import uuid

class UserDocument(models.Model):
    """Model to store user document metadata"""
    
    DOCUMENT_TYPES = [
        ('medical_report', 'Medical Report'),
        ('lab_result', 'Lab Result'),
        ('prescription', 'Prescription'),
        ('imaging', 'Imaging Report'),
        ('other', 'Other'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.CharField(max_length=128, db_index=True)  # Firebase UID
    document_name = models.CharField(max_length=255)
    document_type = models.CharField(max_length=50, choices=DOCUMENT_TYPES, default='other')
    upload_date = models.DateTimeField(auto_now_add=True)
    file_path = models.CharField(max_length=500, blank=True, null=True)  # Optional file storage
    file_size = models.IntegerField(default=0)  # File size in bytes
    vector_ids = models.JSONField(default=list)  # Store Pinecone vector IDs for this document
    extracted_text = models.TextField(blank=True)  # Store extracted text content
    processing_status = models.CharField(max_length=20, default='pending')  # pending, processing, completed, failed
    
    class Meta:
        db_table = 'user_documents'
        indexes = [
            models.Index(fields=['user_id']),
            models.Index(fields=['upload_date']),
            models.Index(fields=['document_type']),
        ]
    
    def __str__(self):
        return f"{self.document_name} - {self.user_id}"

class DocumentChunk(models.Model):
    """Model to store individual text chunks from documents"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(UserDocument, on_delete=models.CASCADE, related_name='chunks')
    chunk_index = models.IntegerField()  # Order of chunk in document
    text_content = models.TextField()
    vector_id = models.CharField(max_length=255, unique=True)  # Pinecone vector ID
    embedding_model = models.CharField(max_length=50, default='text-embedding-ada-002')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'document_chunks'
        indexes = [
            models.Index(fields=['document']),
            models.Index(fields=['vector_id']),
        ]
        unique_together = ['document', 'chunk_index']
    
    def __str__(self):
        return f"Chunk {self.chunk_index} of {self.document.document_name}"

class UserChatSession(models.Model):
    """Model to store user chat sessions"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.CharField(max_length=128, db_index=True)  # Firebase UID
    session_id = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'user_chat_sessions'
        indexes = [
            models.Index(fields=['user_id']),
            models.Index(fields=['session_id']),
            models.Index(fields=['last_activity']),
        ]
    
    def __str__(self):
        return f"Session {self.session_id} - {self.user_id}"

class ChatMessage(models.Model):
    """Model to store individual chat messages"""
    
    MESSAGE_TYPES = [
        ('user', 'User Message'),
        ('assistant', 'Assistant Response'),
        ('system', 'System Message'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(UserChatSession, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict)  # Store additional info like used documents, etc.
    
    class Meta:
        db_table = 'chat_messages'
        indexes = [
            models.Index(fields=['session']),
            models.Index(fields=['timestamp']),
            models.Index(fields=['message_type']),
        ]
    
    def __str__(self):
        return f"{self.message_type} message in {self.session.session_id}"
