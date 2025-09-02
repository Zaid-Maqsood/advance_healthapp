import PyPDF2
import re
import uuid
import numpy as np
from typing import List, Dict, Any, Tuple
from .models import UserDocument, DocumentChunk
from .pinecone_utils import get_pinecone_manager
import openai
from decouple import config
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

openai.api_key = config('OPENAI_API_KEY')

class DocumentProcessor:
    """Process documents and store them in vector database with hybrid semantic chunking"""
    
    def __init__(self):
        self.pinecone_manager = get_pinecone_manager()
        self.chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
        
        # Hybrid chunking configuration
        self.min_chunk_size = 200
        self.max_chunk_size = 1000
        self.overlap_size = 100
        
        # Initialize NLP model for semantic chunking
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
        
        # Medical patterns for rule-based chunking
        self.medical_patterns = self._load_medical_patterns()
        
        # Initialize BM25 for keyword search
        self.bm25_indexes = {}  # Per-user BM25 indexes
        self.bm25_corpora = {}  # Per-user document corpora
        self._initialize_nltk()
    
    def _initialize_nltk(self):
        """Initialize NLTK resources for text processing"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK resources initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK: {e}")
            self.stop_words = set()
    
    def _load_medical_patterns(self) -> Dict[str, str]:
        """Load medical section patterns for rule-based chunking"""
        return {
            'vital_signs': r'(?i)(vital signs?|vitals?|blood pressure|heart rate|temperature|temp|bp|hr)',
            'lab_results': r'(?i)(lab results?|laboratory|blood work|test results?|glucose|cholesterol|hemoglobin)',
            'medications': r'(?i)(medications?|drugs?|prescriptions?|current meds|medication list)',
            'diagnosis': r'(?i)(diagnosis|diagnoses|dx|assessment|impression)',
            'symptoms': r'(?i)(symptoms?|complaints?|chief complaint|presenting complaint)',
            'history': r'(?i)(history|hx|past medical|medical history|family history)',
            'physical_exam': r'(?i)(physical exam|pe|examination|exam findings)',
            'treatment': r'(?i)(treatment|therapy|plan|management|follow.?up)',
            'allergies': r'(?i)(allergies?|allergic|adverse reactions?)',
            'social_history': r'(?i)(social history|smoking|alcohol|drug use)'
        }
    
    def analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure to determine chunking strategy"""
        structure_score = 0.0
        total_checks = 0
        
        # Check for section headers
        if re.search(r'(?i)(vital signs|lab results|medications|diagnosis)', text):
            structure_score += 1
        total_checks += 1
        
        # Check for consistent formatting
        if re.search(r'\n\s*[A-Z][a-z]+:', text):
            structure_score += 1
        total_checks += 1
        
        # Check for bullet points or lists
        if re.search(r'^\s*[-•*]\s', text, re.MULTILINE):
            structure_score += 1
        total_checks += 1
        
        # Check for numbered sections
        if re.search(r'^\s*\d+\.\s', text, re.MULTILINE):
            structure_score += 1
        total_checks += 1
        
        # Check for medical terminology density
        medical_terms = len(re.findall(r'\b(bp|hr|temp|glucose|cholesterol|medication|symptom)\b', text, re.I))
        if medical_terms > 5:
            structure_score += 1
        total_checks += 1
        
        final_score = structure_score / total_checks if total_checks > 0 else 0
        
        return {
            'score': final_score,
            'is_structured': final_score > 0.7,
            'is_semi_structured': 0.3 <= final_score <= 0.7,
            'is_unstructured': final_score < 0.3,
            'medical_terms_count': medical_terms
        }
    
    def detect_medical_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect medical sections using rule-based patterns"""
        boundaries = []
        
        for section_type, pattern in self.medical_patterns.items():
            for match in re.finditer(pattern, text):
                boundaries.append({
                    'start': match.start(),
                    'end': match.end(),
                    'type': section_type,
                    'confidence': 0.9,
                    'text': match.group()
                })
        
        # Sort by position
        boundaries.sort(key=lambda x: x['start'])
        
        # Merge overlapping boundaries
        merged_boundaries = []
        for boundary in boundaries:
            if not merged_boundaries or boundary['start'] > merged_boundaries[-1]['end']:
                merged_boundaries.append(boundary)
            else:
                # Merge with previous boundary
                prev = merged_boundaries[-1]
                prev['end'] = max(prev['end'], boundary['end'])
                prev['type'] = f"{prev['type']}_{boundary['type']}"
        
        return merged_boundaries
    
    def split_into_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sentences with metadata"""
        # Simple sentence splitting - can be enhanced with spaCy
        sentences = re.split(r'[.!?]+', text)
        sentence_objects = []
        
        current_pos = 0
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:
                sentence_objects.append({
                    'text': sentence,
                    'index': i,
                    'start': current_pos,
                    'end': current_pos + len(sentence),
                    'length': len(sentence)
                })
                current_pos += len(sentence) + 1  # +1 for the delimiter
        
        return sentence_objects
    
    def calculate_sentence_similarities(self, embeddings: np.ndarray) -> List[float]:
        """Calculate similarity between adjacent sentences"""
        similarities = []
        
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
        
        return similarities
    
    def find_semantic_boundaries(self, similarities: List[float], threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Find semantic boundaries based on similarity drops"""
        boundaries = []
        
        for i, similarity in enumerate(similarities):
            if similarity < threshold:
                boundaries.append({
                    'position': i + 1,
                    'confidence': 1 - similarity,
                    'similarity': similarity
                })
        
        return boundaries
    
    def detect_content_type(self, text: str) -> str:
        """Detect content type of text chunk"""
        # Vital signs patterns
        if re.search(r'\b\d{2,3}/\d{2,3}\b', text):  # BP pattern
            return 'vital_signs'
        
        # Lab values patterns
        if re.search(r'\b\d+\.?\d*\s*(mg/dl|mmol/l|units?|mg%)\b', text, re.I):
            return 'lab_results'
        
        # Medication patterns
        if re.search(r'\b\d+\s*(mg|ml|tablet|capsule|twice daily|bid|tid)\b', text, re.I):
            return 'medications'
        
        # Symptoms patterns
        if re.search(r'\b(pain|ache|fever|nausea|dizziness|shortness of breath)\b', text, re.I):
            return 'symptoms'
        
        # Temporal patterns
        if re.search(r'\b(follow up|next visit|in \d+ weeks?|appointment)\b', text, re.I):
            return 'follow_up'
        
        return 'general'
    
    def hybrid_semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Main hybrid chunking method that combines rule-based, NLP, and overlap strategies"""
        try:
            # Step 1: Analyze document structure
            structure_analysis = self.analyze_document_structure(text)
            
            # Step 2: Apply appropriate chunking strategy
            if structure_analysis['is_structured']:
                chunks = self.structured_hybrid_chunking(text, structure_analysis)
            elif structure_analysis['is_semi_structured']:
                chunks = self.semi_structured_hybrid_chunking(text, structure_analysis)
            else:
                chunks = self.unstructured_hybrid_chunking(text)
            
            # Step 3: Apply overlap strategy
            overlapped_chunks = self.apply_overlap_strategy(chunks, text)
            
            # Step 4: Enhance with metadata
            enhanced_chunks = self.enhance_chunks_with_metadata(overlapped_chunks, text)
            
            # Step 5: Quality validation
            validated_chunks = self.validate_chunk_quality(enhanced_chunks)
            
            return validated_chunks
            
        except Exception as e:
            logger.error(f"Hybrid chunking failed: {e}")
            # Fallback to basic chunking
            return self.fallback_basic_chunking(text)
    
    def structured_hybrid_chunking(self, text: str, structure_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Hybrid chunking for well-structured medical documents"""
        chunks = []
        
        # Step 1: Rule-based section detection
        section_boundaries = self.detect_medical_sections(text)
        
        if not section_boundaries:
            # No clear sections found, use NLP chunking
            return self.nlp_semantic_chunking(text)
        
        # Step 2: Process each section
        for i, section in enumerate(section_boundaries):
            start = section['start']
            end = section['end'] if i < len(section_boundaries) - 1 else len(text)
            section_text = text[start:end]
            
            # Step 3: Apply NLP within sections
            section_chunks = self.nlp_chunk_within_section(section_text, section['type'])
            
            # Step 4: Add section metadata
            for chunk in section_chunks:
                chunk['section_type'] = section['type']
                chunk['section_start'] = start
                chunk['confidence'] = 0.9  # High confidence for structured content
                chunk['chunk_method'] = 'structured_hybrid'
            
            chunks.extend(section_chunks)
        
        return chunks
    
    def semi_structured_hybrid_chunking(self, text: str, structure_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Hybrid chunking for semi-structured documents"""
        chunks = []
        
        # Try to find some structure first
        section_boundaries = self.detect_medical_sections(text)
        
        if section_boundaries:
            # Use structured approach for found sections
            structured_chunks = self.structured_hybrid_chunking(text, structure_analysis)
            chunks.extend(structured_chunks)
        else:
            # Use content-aware chunking
            content_chunks = self.content_aware_chunking(text)
            chunks.extend(content_chunks)
        
        # Apply NLP semantic analysis to all chunks
        semantic_chunks = self.apply_semantic_analysis(chunks)
        
        return semantic_chunks
    
    def unstructured_hybrid_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Hybrid chunking for unstructured medical documents"""
        chunks = []
        
        # Step 1: Content-aware chunking
        content_chunks = self.content_aware_chunking(text)
        
        # Step 2: Apply NLP semantic analysis
        semantic_chunks = self.apply_semantic_analysis(content_chunks)
        
        # Step 3: Merge and optimize
        final_chunks = self.merge_and_optimize_chunks(semantic_chunks)
        
        return final_chunks
    
    def nlp_semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
        """NLP-based semantic chunking"""
        if not self.sentence_model:
            return self.fallback_basic_chunking(text)
        
        try:
            # Split into sentences
            sentences = self.split_into_sentences(text)
            
            if len(sentences) <= 2:
                return [{
                    'text': text,
                    'type': 'general',
                    'sentences': sentences,
                    'chunk_method': 'single_section',
                    'confidence': 0.8
                }]
            
            # Generate embeddings
            sentence_texts = [s['text'] for s in sentences]
            embeddings = self.sentence_model.encode(sentence_texts)
            
            # Calculate similarities
            similarities = self.calculate_sentence_similarities(embeddings)
            
            # Find semantic boundaries
            boundaries = self.find_semantic_boundaries(similarities, threshold=0.7)
            
            # Create chunks
            chunks = []
            start_idx = 0
            
            for boundary in boundaries:
                end_idx = boundary['position']
                chunk_sentences = sentences[start_idx:end_idx]
                chunk_text = ' '.join([s['text'] for s in chunk_sentences])
                
                chunks.append({
                    'text': chunk_text,
                    'type': self.detect_content_type(chunk_text),
                    'sentences': chunk_sentences,
                    'chunk_method': 'nlp_semantic',
                    'boundary_confidence': boundary['confidence'],
                    'confidence': 0.8
                })
                
                start_idx = end_idx
            
            # Add remaining sentences
            if start_idx < len(sentences):
                chunk_sentences = sentences[start_idx:]
                chunk_text = ' '.join([s['text'] for s in chunk_sentences])
                chunks.append({
                    'text': chunk_text,
                    'type': self.detect_content_type(chunk_text),
                    'sentences': chunk_sentences,
                    'chunk_method': 'nlp_remaining',
                    'confidence': 0.8
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"NLP semantic chunking failed: {e}")
            return self.fallback_basic_chunking(text)
    
    def nlp_chunk_within_section(self, section_text: str, section_type: str) -> List[Dict[str, Any]]:
        """Apply NLP chunking within a medical section"""
        if not self.sentence_model:
            return [{
                'text': section_text,
                'type': section_type,
                'chunk_method': 'section_fallback',
                'confidence': 0.7
            }]
        
        try:
            # Split into sentences
            sentences = self.split_into_sentences(section_text)
            
            if len(sentences) <= 2:
                return [{
                    'text': section_text,
                    'type': section_type,
                    'sentences': sentences,
                    'chunk_method': 'single_section',
                    'confidence': 0.9
                }]
            
            # Generate embeddings
            sentence_texts = [s['text'] for s in sentences]
            embeddings = self.sentence_model.encode(sentence_texts)
            
            # Calculate similarities
            similarities = self.calculate_sentence_similarities(embeddings)
            
            # Find semantic boundaries
            boundaries = self.find_semantic_boundaries(similarities, threshold=0.7)
            
            # Create chunks
            chunks = []
            start_idx = 0
            
            for boundary in boundaries:
                end_idx = boundary['position']
                chunk_sentences = sentences[start_idx:end_idx]
                chunk_text = ' '.join([s['text'] for s in chunk_sentences])
                
                chunks.append({
                    'text': chunk_text,
                    'type': section_type,
                    'sentences': chunk_sentences,
                    'chunk_method': 'section_semantic',
                    'boundary_confidence': boundary['confidence'],
                    'confidence': 0.9
                })
                
                start_idx = end_idx
            
            # Add remaining sentences
            if start_idx < len(sentences):
                chunk_sentences = sentences[start_idx:]
                chunk_text = ' '.join([s['text'] for s in chunk_sentences])
                chunks.append({
                    'text': chunk_text,
                    'type': section_type,
                    'sentences': chunk_sentences,
                    'chunk_method': 'section_remaining',
                    'confidence': 0.9
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Section NLP chunking failed: {e}")
            return [{
                'text': section_text,
                'type': section_type,
                'chunk_method': 'section_fallback',
                'confidence': 0.7
            }]
    
    def content_aware_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Chunk based on content type and medical entities"""
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_type = 'general'
        
        for sentence in sentences:
            # Detect content type
            sentence_type = self.detect_content_type(sentence['text'])
            
            # Decide whether to continue current chunk or start new one
            if self.should_continue_chunk(current_type, sentence_type):
                current_chunk.append(sentence)
                current_type = sentence_type
            else:
                # Create chunk from current content
                if current_chunk:
                    chunks.append(self.create_chunk_from_sentences(current_chunk, current_type))
                
                # Start new chunk
                current_chunk = [sentence]
                current_type = sentence_type
        
        # Add final chunk
        if current_chunk:
            chunks.append(self.create_chunk_from_sentences(current_chunk, current_type))
        
        return chunks
    
    def should_continue_chunk(self, current_type: str, sentence_type: str) -> bool:
        """Determine if sentence should continue current chunk"""
        # Same type - continue
        if current_type == sentence_type:
            return True
        
        # Related types - continue
        related_types = {
            'vital_signs': ['lab_results'],
            'lab_results': ['vital_signs'],
            'symptoms': ['diagnosis'],
            'diagnosis': ['symptoms', 'treatment'],
            'medications': ['treatment']
        }
        
        if current_type in related_types and sentence_type in related_types[current_type]:
            return True
        
        # General type - continue with most content types
        if current_type == 'general' and sentence_type != 'general':
            return True
        
        return False
    
    def create_chunk_from_sentences(self, sentences: List[Dict[str, Any]], chunk_type: str) -> Dict[str, Any]:
        """Create chunk from sentence list"""
        chunk_text = ' '.join([s['text'] for s in sentences])
        
        return {
            'text': chunk_text,
            'type': chunk_type,
            'sentences': sentences,
            'chunk_method': 'content_aware',
            'confidence': 0.8
        }
    
    def apply_semantic_analysis(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply NLP semantic analysis to content chunks"""
        if not self.sentence_model:
            return chunks
        
        enhanced_chunks = []
        
        for chunk in chunks:
            # If chunk is too large, apply semantic splitting
            if len(chunk['text']) > self.max_chunk_size:
                sub_chunks = self.semantic_split_large_chunk(chunk)
                enhanced_chunks.extend(sub_chunks)
            else:
                # Enhance existing chunk with semantic metadata
                enhanced_chunk = self.enhance_chunk_with_semantics(chunk)
                enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def semantic_split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split large chunks using semantic analysis"""
        if not self.sentence_model:
            return [chunk]
        
        try:
            sentences = chunk['sentences']
            sentence_texts = [s['text'] for s in sentences]
            embeddings = self.sentence_model.encode(sentence_texts)
            
            # Find semantic boundaries
            similarities = self.calculate_sentence_similarities(embeddings)
            boundaries = self.find_semantic_boundaries(similarities, threshold=0.6)
            
            # Create sub-chunks
            sub_chunks = []
            start_idx = 0
            
            for boundary in boundaries:
                end_idx = boundary['position']
                sub_sentences = sentences[start_idx:end_idx]
                sub_text = ' '.join([s['text'] for s in sub_sentences])
                
                sub_chunks.append({
                    'text': sub_text,
                    'type': chunk['type'],
                    'sentences': sub_sentences,
                    'chunk_method': 'semantic_split',
                    'parent_chunk_id': chunk.get('id'),
                    'boundary_confidence': boundary['confidence'],
                    'confidence': 0.8
                })
                
                start_idx = end_idx
            
            # Add remaining sentences
            if start_idx < len(sentences):
                sub_sentences = sentences[start_idx:]
                sub_text = ' '.join([s['text'] for s in sub_sentences])
                sub_chunks.append({
                    'text': sub_text,
                    'type': chunk['type'],
                    'sentences': sub_sentences,
                    'chunk_method': 'semantic_split_remaining',
                    'parent_chunk_id': chunk.get('id'),
                    'confidence': 0.8
                })
            
            return sub_chunks
            
        except Exception as e:
            logger.error(f"Semantic splitting failed: {e}")
            return [chunk]
    
    def enhance_chunk_with_semantics(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance chunk with semantic metadata"""
        enhanced_chunk = chunk.copy()
        
        # Add semantic features
        enhanced_chunk['semantic_features'] = {
            'sentence_count': len(chunk.get('sentences', [])),
            'avg_sentence_length': np.mean([s['length'] for s in chunk.get('sentences', [])]) if chunk.get('sentences') else 0,
            'content_density': len(chunk['text']) / max(1, len(chunk.get('sentences', []))),
            'medical_terms_count': len(re.findall(r'\b(bp|hr|temp|glucose|cholesterol|medication|symptom|diagnosis)\b', chunk['text'], re.I))
        }
        
        return enhanced_chunk
    
    def merge_and_optimize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge and optimize chunks for better quality"""
        optimized_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Check if chunk is too small and can be merged with next
            if (len(current_chunk['text']) < self.min_chunk_size and 
                i < len(chunks) - 1 and 
                current_chunk['type'] == chunks[i + 1]['type']):
                
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged_text = current_chunk['text'] + ' ' + next_chunk['text']
                merged_sentences = current_chunk.get('sentences', []) + next_chunk.get('sentences', [])
                
                optimized_chunks.append({
                    'text': merged_text,
                    'type': current_chunk['type'],
                    'sentences': merged_sentences,
                    'chunk_method': 'merged',
                    'confidence': min(current_chunk.get('confidence', 0.8), next_chunk.get('confidence', 0.8))
                })
                
                i += 2  # Skip next chunk as it's merged
            else:
                optimized_chunks.append(current_chunk)
                i += 1
        
        return optimized_chunks
    
    def apply_overlap_strategy(self, chunks: List[Dict[str, Any]], original_text: str) -> List[Dict[str, Any]]:
        """Apply intelligent overlap between chunks"""
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Calculate overlap based on chunk type and content
            overlap_size = self.calculate_optimal_overlap(chunk, i, chunks)
            
            # Add overlap from previous chunk
            if i > 0 and overlap_size > 0:
                prev_chunk = chunks[i-1]
                overlap_text = self.extract_overlap_text(prev_chunk, overlap_size)
                chunk['text'] = overlap_text + ' ' + chunk['text']
                chunk['overlap_from_previous'] = overlap_text
                chunk['overlap_size'] = len(overlap_text)
            
            overlapped_chunks.append(chunk)
        
        return overlapped_chunks
    
    def calculate_optimal_overlap(self, chunk: Dict[str, Any], index: int, all_chunks: List[Dict[str, Any]]) -> int:
        """Calculate optimal overlap size based on content type and context"""
        base_overlap = self.overlap_size
        
        # Adjust based on chunk type
        if chunk['type'] in ['lab_results', 'medications']:
            # Medical data needs more context
            return int(base_overlap * 1.5)
        elif chunk['type'] in ['symptoms', 'diagnosis']:
            # Clinical information needs context
            return int(base_overlap * 1.2)
        elif chunk['type'] == 'general':
            # General text needs less overlap
            return int(base_overlap * 0.8)
        
        # Adjust based on position
        if index == 0 or index == len(all_chunks) - 1:
            # First or last chunk needs less overlap
            return int(base_overlap * 0.5)
        
        return base_overlap
    
    def extract_overlap_text(self, chunk: Dict[str, Any], overlap_size: int) -> str:
        """Extract overlap text from chunk"""
        text = chunk['text']
        
        if len(text) <= overlap_size:
            return text
        
        # Try to break at sentence boundary
        sentences = chunk.get('sentences', [])
        if sentences:
            overlap_text = ""
            for sentence in reversed(sentences):
                if len(overlap_text + sentence['text']) <= overlap_size:
                    overlap_text = sentence['text'] + ' ' + overlap_text
                else:
                    break
            return overlap_text.strip()
        
        # Fallback to character-based overlap
        return text[-overlap_size:].strip()
    
    def enhance_chunks_with_metadata(self, chunks: List[Dict[str, Any]], original_text: str) -> List[Dict[str, Any]]:
        """Enhance chunks with additional metadata"""
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            enhanced_chunk = chunk.copy()
            
            # Add metadata
            enhanced_chunk['metadata'] = {
                'chunk_index': i,
                'chunk_method': chunk.get('chunk_method', 'hybrid'),
                'content_type': chunk['type'],
                'confidence': chunk.get('confidence', 0.8),
                'sentence_count': len(chunk.get('sentences', [])),
                'character_count': len(chunk['text']),
                'overlap_size': chunk.get('overlap_size', 0),
                'boundary_confidence': chunk.get('boundary_confidence', 0.8),
                'section_type': chunk.get('section_type'),
                'medical_terms': self.extract_medical_terms(chunk['text'])
            }
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def extract_medical_terms(self, text: str) -> List[str]:
        """Extract medical terms from text"""
        medical_terms = []
        
        # Common medical terms patterns
        patterns = [
            r'\b\d{2,3}/\d{2,3}\b',  # BP
            r'\b\d+\.?\d*\s*(mg/dl|mmol/l|units?|mg%)\b',  # Lab values
            r'\b\d+\s*(mg|ml|tablet|capsule)\b',  # Medications
            r'\b(bp|hr|temp|glucose|cholesterol|hemoglobin)\b',  # Medical terms
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.I)
            medical_terms.extend(matches)
        
        return list(set(medical_terms))  # Remove duplicates
    
    def validate_chunk_quality(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and enhance chunk quality"""
        validated_chunks = []
        
        for chunk in chunks:
            # Quality checks
            quality_score = self.calculate_chunk_quality(chunk)
            
            if quality_score > 0.6:
                # High quality chunk
                chunk['quality_score'] = quality_score
                chunk['validation_status'] = 'approved'
                validated_chunks.append(chunk)
            elif quality_score > 0.3:
                # Medium quality - enhance
                enhanced_chunk = self.enhance_chunk_quality(chunk)
                enhanced_chunk['quality_score'] = quality_score
                enhanced_chunk['validation_status'] = 'enhanced'
                validated_chunks.append(enhanced_chunk)
            else:
                # Low quality - try to fix
                fixed_chunk = self.fix_low_quality_chunk(chunk)
                if fixed_chunk:
                    fixed_chunk['quality_score'] = quality_score
                    fixed_chunk['validation_status'] = 'fixed'
                    validated_chunks.append(fixed_chunk)
        
        return validated_chunks
    
    def calculate_chunk_quality(self, chunk: Dict[str, Any]) -> float:
        """Calculate quality score for a chunk"""
        score = 0.0
        
        # Coherence score
        coherence = self.calculate_coherence(chunk)
        score += coherence * 0.3
        
        # Completeness score
        completeness = self.calculate_completeness(chunk)
        score += completeness * 0.2
        
        # Medical accuracy score
        medical_accuracy = self.calculate_medical_accuracy(chunk)
        score += medical_accuracy * 0.3
        
        # Size appropriateness score
        size_score = self.calculate_size_score(chunk)
        score += size_score * 0.2
        
        return score
    
    def calculate_coherence(self, chunk: Dict[str, Any]) -> float:
        """Calculate semantic coherence within chunk"""
        sentences = chunk.get('sentences', [])
        if len(sentences) < 2:
            return 1.0
        
        if not self.sentence_model:
            return 0.8  # Default score if no model
        
        try:
            sentence_texts = [s['text'] for s in sentences]
            embeddings = self.sentence_model.encode(sentence_texts)
            similarities = self.calculate_sentence_similarities(embeddings)
            return np.mean(similarities) if similarities else 0.8
        except:
            return 0.8
    
    def calculate_completeness(self, chunk: Dict[str, Any]) -> float:
        """Calculate completeness score"""
        text = chunk['text']
        
        # Check for incomplete sentences
        if text.endswith(('and', 'or', 'but', 'the', 'a', 'an')):
            return 0.6
        
        # Check for proper sentence endings
        if re.search(r'[.!?]$', text.strip()):
            return 1.0
        
        return 0.8
    
    def calculate_medical_accuracy(self, chunk: Dict[str, Any]) -> float:
        """Calculate medical accuracy score"""
        text = chunk['text']
        medical_terms = self.extract_medical_terms(text)
        
        # Higher score for more medical terms
        if len(medical_terms) > 3:
            return 1.0
        elif len(medical_terms) > 1:
            return 0.8
        else:
            return 0.6
    
    def calculate_size_score(self, chunk: Dict[str, Any]) -> float:
        """Calculate size appropriateness score"""
        text_length = len(chunk['text'])
        
        if self.min_chunk_size <= text_length <= self.max_chunk_size:
            return 1.0
        elif text_length < self.min_chunk_size:
            return 0.6
        else:
            return 0.8
    
    def enhance_chunk_quality(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance chunk quality"""
        enhanced_chunk = chunk.copy()
        
        # Add quality enhancements
        enhanced_chunk['quality_enhancements'] = {
            'coherence_boost': True,
            'metadata_enriched': True,
            'medical_terms_highlighted': True
        }
        
        return enhanced_chunk
    
    def fix_low_quality_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Try to fix low quality chunk"""
        # For now, just return the chunk with a note
        fixed_chunk = chunk.copy()
        fixed_chunk['quality_notes'] = 'Low quality chunk - may need manual review'
        return fixed_chunk
    
    def fallback_basic_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Fallback to basic chunking if hybrid methods fail"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period > start + self.chunk_size * 0.7:
                    end = last_period + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'type': self.detect_content_type(chunk_text),
                    'chunk_method': 'fallback_basic',
                    'confidence': 0.5,
                    'metadata': {
                        'chunk_index': len(chunks),
                        'fallback_reason': 'hybrid_methods_failed'
                    }
                })
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
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
        """Split text into overlapping chunks using hybrid semantic chunking"""
        try:
            # Use hybrid semantic chunking
            hybrid_chunks = self.hybrid_semantic_chunking(text)
            
            # Extract just the text content for backward compatibility
            text_chunks = [chunk['text'] for chunk in hybrid_chunks]
            
            logger.info(f"Created {len(text_chunks)} chunks using hybrid semantic chunking")
            return text_chunks
            
        except Exception as e:
            logger.error(f"Hybrid chunking failed, falling back to basic chunking: {e}")
            return self.create_basic_text_chunks(text)
    
    def create_basic_text_chunks(self, text: str) -> List[str]:
        """Fallback basic chunking method"""
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
                from openai import OpenAI
                client = OpenAI(api_key=config('OPENAI_API_KEY'))
                
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                raise
        
        return embeddings
    
    def store_document_vectors(self, user_id: str, document_name: str, 
                             chunks: List[str], embeddings: List[List[float]], 
                             document_type: str = 'other', hybrid_chunks: List[Dict[str, Any]] = None) -> UserDocument:
        """Store document chunks in Pinecone and create database records with enhanced metadata"""
        
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
                
                # Get enhanced metadata if available
                enhanced_metadata = {}
                if hybrid_chunks and i < len(hybrid_chunks):
                    hybrid_chunk = hybrid_chunks[i]
                    enhanced_metadata = {
                        "chunk_method": hybrid_chunk.get('chunk_method', 'hybrid'),
                        "content_type": hybrid_chunk.get('type', 'general'),
                        "confidence": hybrid_chunk.get('confidence', 0.8),
                        "quality_score": hybrid_chunk.get('quality_score', 0.8),
                        "validation_status": hybrid_chunk.get('validation_status', 'approved'),
                        "sentence_count": hybrid_chunk.get('metadata', {}).get('sentence_count', 0),
                        "overlap_size": hybrid_chunk.get('overlap_size', 0),
                        "boundary_confidence": hybrid_chunk.get('boundary_confidence', 0.8),
                        "section_type": hybrid_chunk.get('section_type'),
                        "medical_terms": hybrid_chunk.get('metadata', {}).get('medical_terms', [])
                    }
                
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
                        "source": "user_document",
                        **enhanced_metadata  # Add enhanced metadata
                    }
                }
                vectors.append(vector_data)
                
                # Create chunk record with enhanced metadata
                chunk_record = DocumentChunk(
                    document=document,
                    chunk_index=i,
                    text_content=chunk,
                    vector_id=vector_id,
                    metadata=enhanced_metadata  # Store enhanced metadata
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
                
                logger.info(f"Successfully stored document {document.id} with {len(chunks)} chunks using hybrid chunking")
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
        """Complete document processing pipeline with hybrid semantic chunking"""
        
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_file)
            
            # Create hybrid semantic chunks
            hybrid_chunks = self.hybrid_semantic_chunking(text)
            
            # Extract text chunks for embeddings
            chunks = [chunk['text'] for chunk in hybrid_chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # Store in vector database with enhanced metadata
            document = self.store_document_vectors(
                user_id=user_id,
                document_name=document_name,
                chunks=chunks,
                embeddings=embeddings,
                document_type=document_type,
                hybrid_chunks=hybrid_chunks  # Pass hybrid chunks for metadata
            )
            
            logger.info(f"Successfully processed document {document_name} with hybrid semantic chunking")
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
    
    def _preprocess_text_for_bm25(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing"""
        try:
            # Tokenize and lowercase
            tokens = word_tokenize(text.lower())
            # Remove stopwords and non-alphabetic tokens
            tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
            return tokens
        except Exception as e:
            logger.warning(f"Error preprocessing text for BM25: {e}")
            return text.lower().split()
    
    def _ensure_bm25_initialized(self, user_id: str):
        """Ensure BM25 index is initialized for user"""
        if user_id not in self.bm25_indexes:
            # Get all user documents
            documents = self.get_user_documents(user_id)
            corpus = []
            doc_metadata = []
            
            for doc in documents:
                # Get all chunks for this document
                chunks = DocumentChunk.objects.filter(document=doc).order_by('chunk_index')
                for chunk in chunks:
                    processed_text = self._preprocess_text_for_bm25(chunk.text_content)
                    if processed_text:  # Only add non-empty processed text
                        corpus.append(processed_text)
                        doc_metadata.append({
                            'chunk_id': str(chunk.id),
                            'document_id': str(doc.id),
                            'document_name': doc.document_name,
                            'document_type': doc.document_type,
                            'chunk_index': chunk.chunk_index,
                            'text_content': chunk.text_content
                        })
            
            if corpus:
                self.bm25_indexes[user_id] = BM25Okapi(corpus)
                self.bm25_corpora[user_id] = doc_metadata
                logger.info(f"Initialized BM25 index for user {user_id} with {len(corpus)} documents")
            else:
                self.bm25_indexes[user_id] = None
                self.bm25_corpora[user_id] = []
                logger.info(f"No documents found for user {user_id} to initialize BM25")
    
    def _keyword_search_bm25(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Perform BM25 keyword search for user's documents"""
        try:
            # Ensure BM25 is initialized for this user
            self._ensure_bm25_initialized(user_id)
            
            if user_id not in self.bm25_indexes or self.bm25_indexes[user_id] is None:
                logger.warning(f"No BM25 index available for user {user_id}")
                return []
            
            # Preprocess query
            query_tokens = self._preprocess_text_for_bm25(query)
            if not query_tokens:
                logger.warning(f"No valid tokens in query: {query}")
                return []
            
            # Get BM25 scores
            bm25_scores = self.bm25_indexes[user_id].get_scores(query_tokens)
            
            # Create results with scores and metadata
            results = []
            for i, score in enumerate(bm25_scores):
                if score > 0:  # Only include documents with positive scores
                    metadata = self.bm25_corpora[user_id][i]
                    results.append({
                        'id': metadata['chunk_id'],
                        'score': float(score),
                        'text_content': metadata['text_content'],
                        'document_name': metadata['document_name'],
                        'document_type': metadata['document_type'],
                        'chunk_index': metadata['chunk_index'],
                        'metadata': metadata,
                        'search_type': 'bm25'
                    })
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in BM25 keyword search: {str(e)}")
            return []
    
    def _analyze_query_characteristics(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine optimal search weights"""
        try:
            # Preprocess query for analysis
            query_tokens = self._preprocess_text_for_bm25(query)
            query_lower = query.lower()
            
            # Medical terminology indicators (favor semantic search)
            medical_terms = [
                'symptoms', 'diagnosis', 'treatment', 'medication', 'allergy', 'allergies',
                'condition', 'disease', 'syndrome', 'disorder', 'chronic', 'acute',
                'therapy', 'procedure', 'surgery', 'examination', 'test', 'results',
                'vital', 'signs', 'blood', 'pressure', 'heart', 'rate', 'temperature',
                'glucose', 'cholesterol', 'hemoglobin', 'diabetes', 'hypertension'
            ]
            
            # Specific medical terms (favor keyword search)
            specific_terms = [
                'mg', 'ml', 'units', 'dose', 'dosage', 'tablet', 'capsule', 'injection',
                'bpm', 'mmhg', 'fahrenheit', 'celsius', 'normal', 'high', 'low',
                'positive', 'negative', 'abnormal', 'elevated', 'decreased'
            ]
            
            # Count medical terminology
            medical_count = sum(1 for term in medical_terms if term in query_lower)
            specific_count = sum(1 for term in specific_terms if term in query_lower)
            
            # Query length analysis
            query_length = len(query_tokens)
            is_short_query = query_length <= 3
            is_long_query = query_length >= 8
            
            # Question type analysis
            is_question = query.strip().endswith('?')
            has_question_words = any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'which'])
            
            # Determine optimal weights
            if medical_count >= 2 and not specific_count:
                # Medical concept queries - favor semantic search
                semantic_weight = 0.8
                keyword_weight = 0.2
                strategy = "medical_concept"
            elif specific_count >= 1 or (medical_count >= 1 and specific_count >= 1):
                # Specific medical data - balance both
                semantic_weight = 0.6
                keyword_weight = 0.4
                strategy = "specific_medical"
            elif is_short_query and not is_question:
                # Short queries - favor keyword search
                semantic_weight = 0.4
                keyword_weight = 0.6
                strategy = "short_query"
            elif is_long_query and (is_question or has_question_words):
                # Complex questions - favor semantic search
                semantic_weight = 0.8
                keyword_weight = 0.2
                strategy = "complex_question"
            else:
                # Default balanced approach
                semantic_weight = 0.7
                keyword_weight = 0.3
                strategy = "balanced"
            
            analysis = {
                'query_tokens': query_tokens,
                'query_length': query_length,
                'medical_count': medical_count,
                'specific_count': specific_count,
                'is_short_query': is_short_query,
                'is_long_query': is_long_query,
                'is_question': is_question,
                'has_question_words': has_question_words,
                'semantic_weight': semantic_weight,
                'keyword_weight': keyword_weight,
                'strategy': strategy
            }
            
            logger.info(f"Query analysis: '{query}' -> {strategy} (semantic: {semantic_weight}, keyword: {keyword_weight})")
            return analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing query characteristics: {e}")
            # Return default weights
            return {
                'semantic_weight': 0.7,
                'keyword_weight': 0.3,
                'strategy': 'default_fallback'
            }
    
    def _hybrid_search(self, user_id: str, query: str, top_k: int = 5, 
                      semantic_weight: float = None, keyword_weight: float = None) -> List[Dict]:
        """Perform hybrid search combining semantic and BM25 results with intelligent weight adjustment"""
        try:
            # Analyze query to determine optimal weights if not provided
            if semantic_weight is None or keyword_weight is None:
                query_analysis = self._analyze_query_characteristics(query)
                semantic_weight = query_analysis['semantic_weight']
                keyword_weight = query_analysis['keyword_weight']
                strategy = query_analysis['strategy']
            else:
                strategy = "manual_weights"
            
            logger.info(f"Using hybrid search strategy: {strategy} (semantic: {semantic_weight}, keyword: {keyword_weight})")
            
            # Get semantic search results
            semantic_results = self.search_user_documents(user_id, query, top_k * 2)  # Get more for fusion
            
            # Get BM25 keyword search results
            bm25_results = self._keyword_search_bm25(user_id, query, top_k * 2)  # Get more for fusion
            
            # Add search type to semantic results
            for result in semantic_results:
                result['search_type'] = 'semantic'
            
            # Normalize scores and combine with intelligent weights
            combined_results = self._fuse_search_results(semantic_results, bm25_results, 
                                                       semantic_weight, keyword_weight)
            
            # Add strategy info to results for debugging
            for result in combined_results:
                result['search_strategy'] = strategy
                result['weights_used'] = {'semantic': semantic_weight, 'keyword': keyword_weight}
            
            # Return top_k results
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            # Fallback to semantic search only
            logger.info("Falling back to semantic search only")
            return self.search_user_documents(user_id, query, top_k)
    
    def _fuse_search_results(self, semantic_results: List[Dict], bm25_results: List[Dict],
                           semantic_weight: float, keyword_weight: float) -> List[Dict]:
        """Fuse semantic and BM25 search results using score normalization and weighting"""
        try:
            # Create a dictionary to store combined results
            combined_results = {}
            
            # Normalize semantic scores (min-max normalization)
            if semantic_results:
                semantic_scores = [r['score'] for r in semantic_results]
                min_semantic = min(semantic_scores)
                max_semantic = max(semantic_scores)
                semantic_range = max_semantic - min_semantic if max_semantic > min_semantic else 1
                
                for result in semantic_results:
                    normalized_score = (result['score'] - min_semantic) / semantic_range
                    result['normalized_semantic_score'] = normalized_score
                    combined_results[result['id']] = result
            
            # Normalize BM25 scores (min-max normalization)
            if bm25_results:
                bm25_scores = [r['score'] for r in bm25_results]
                min_bm25 = min(bm25_scores)
                max_bm25 = max(bm25_scores)
                bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1
                
                for result in bm25_results:
                    normalized_score = (result['score'] - min_bm25) / bm25_range
                    result['normalized_bm25_score'] = normalized_score
                    
                    if result['id'] in combined_results:
                        # Combine scores
                        combined_results[result['id']]['normalized_bm25_score'] = normalized_score
                    else:
                        # Add new result
                        combined_results[result['id']] = result
            
            # Calculate final hybrid scores
            final_results = []
            for result_id, result in combined_results.items():
                semantic_score = result.get('normalized_semantic_score', 0)
                bm25_score = result.get('normalized_bm25_score', 0)
                
                # Weighted combination
                hybrid_score = (semantic_weight * semantic_score) + (keyword_weight * bm25_score)
                
                result['hybrid_score'] = hybrid_score
                result['search_type'] = 'hybrid'
                final_results.append(result)
            
            # Sort by hybrid score
            final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error fusing search results: {str(e)}")
            # Fallback: return semantic results if available, otherwise BM25
            return semantic_results if semantic_results else bm25_results
    
    def get_most_relevant_medical_data(self, user_id: str, top_k: int = 5) -> List[Dict]:
        """Get most relevant medical data for image-only queries"""
        try:
            # Use a comprehensive medical query to get diverse medical information
            medical_queries = [
                "allergies medications conditions symptoms",
                "vital signs blood pressure heart rate temperature",
                "lab results blood work test results",
                "diagnosis treatment plan medications",
                "medical history chronic conditions"
            ]
            
            all_results = []
            for query in medical_queries:
                results = self._hybrid_search(user_id, query, top_k=2)
                all_results.extend(results)
            
            # Remove duplicates and sort by relevance
            seen_ids = set()
            unique_results = []
            for result in all_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    unique_results.append(result)
            
            # Sort by hybrid score and return top_k
            unique_results.sort(key=lambda x: x.get('hybrid_score', x.get('score', 0)), reverse=True)
            return unique_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting most relevant medical data: {str(e)}")
            # Fallback to simple semantic search
            return self.search_user_documents(user_id, "medical information", top_k)
    
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


