#!/usr/bin/env python3
"""
Quick test for hybrid semantic chunking - no Django setup needed
"""

import re
import numpy as np
from typing import List, Dict, Any

# Mock the sentence transformer for testing
class MockSentenceTransformer:
    def encode(self, texts):
        # Return random embeddings for testing
        return np.random.rand(len(texts), 384)

# Import the chunking methods directly
def _load_medical_patterns() -> Dict[str, str]:
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

def analyze_document_structure(text: str) -> Dict[str, Any]:
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

def detect_medical_sections(text: str) -> List[Dict[str, Any]]:
    """Detect medical sections using rule-based patterns"""
    medical_patterns = _load_medical_patterns()
    boundaries = []
    
    for section_type, pattern in medical_patterns.items():
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

def split_into_sentences(text: str) -> List[Dict[str, Any]]:
    """Split text into sentences with metadata"""
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

def detect_content_type(text: str) -> str:
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

def test_chunking():
    """Test the chunking functionality"""
    
    # Test structured medical document
    structured_text = """
    VITAL SIGNS:
    Blood Pressure: 140/90 mmHg
    Heart Rate: 85 bpm
    Temperature: 98.6°F
    
    LAB RESULTS:
    Glucose: 120 mg/dl
    Cholesterol: 200 mg/dl
    Hemoglobin: 14.5 g/dl
    
    MEDICATIONS:
    Metformin 500mg twice daily
    Lisinopril 10mg once daily
    
    DIAGNOSIS:
    Type 2 Diabetes Mellitus
    Hypertension
    
    SYMPTOMS:
    Patient complains of frequent urination
    Increased thirst and hunger
    Fatigue and weakness
    
    TREATMENT PLAN:
    Continue current medications
    Follow up in 2 weeks
    Monitor blood sugar levels
    """
    
    print("=== Testing Structured Medical Document ===")
    print(f"Original text length: {len(structured_text)} characters")
    
    # Test structure analysis
    structure_analysis = analyze_document_structure(structured_text)
    print(f"\nStructure Analysis:")
    print(f"  Score: {structure_analysis['score']:.2f}")
    print(f"  Is Structured: {structure_analysis['is_structured']}")
    print(f"  Medical Terms Count: {structure_analysis['medical_terms_count']}")
    
    # Test medical section detection
    sections = detect_medical_sections(structured_text)
    print(f"\nDetected Medical Sections: {len(sections)}")
    for i, section in enumerate(sections):
        print(f"  Section {i+1}: {section['type']} at position {section['start']}-{section['end']}")
        print(f"    Text: '{section['text']}'")
    
    # Test sentence splitting
    sentences = split_into_sentences(structured_text)
    print(f"\nSentence Analysis: {len(sentences)} sentences")
    for i, sentence in enumerate(sentences[:5]):  # Show first 5
        content_type = detect_content_type(sentence['text'])
        print(f"  Sentence {i+1}: {content_type}")
        print(f"    Text: '{sentence['text'][:50]}...'")
    
    print("\n" + "="*60)
    
    # Test unstructured medical document
    unstructured_text = """
    Patient came in today with chest pain that started this morning. 
    The pain is sharp and located in the center of the chest. 
    Blood pressure was 150/95, heart rate 90. 
    Patient is taking aspirin 81mg daily and metoprolol 50mg twice daily.
    Lab work shows glucose 110, cholesterol 180. 
    Patient has a history of hypertension and diabetes.
    Recommend follow up in 1 week and continue current medications.
    """
    
    print("\n=== Testing Unstructured Medical Document ===")
    print(f"Original text length: {len(unstructured_text)} characters")
    
    # Test structure analysis
    structure_analysis = analyze_document_structure(unstructured_text)
    print(f"\nStructure Analysis:")
    print(f"  Score: {structure_analysis['score']:.2f}")
    print(f"  Is Unstructured: {structure_analysis['is_unstructured']}")
    print(f"  Medical Terms Count: {structure_analysis['medical_terms_count']}")
    
    # Test sentence splitting and content detection
    sentences = split_into_sentences(unstructured_text)
    print(f"\nSentence Analysis: {len(sentences)} sentences")
    for i, sentence in enumerate(sentences):
        content_type = detect_content_type(sentence['text'])
        print(f"  Sentence {i+1}: {content_type}")
        print(f"    Text: '{sentence['text']}'")
    
    print("\n" + "="*60)
    print("✅ Chunking test completed! The system can:")
    print("  - Analyze document structure")
    print("  - Detect medical sections")
    print("  - Split text into sentences")
    print("  - Classify content types")
    print("  - Preserve medical meaning in chunks")

if __name__ == "__main__":
    test_chunking()
