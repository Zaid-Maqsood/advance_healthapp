#!/usr/bin/env python3
"""
Test script for hybrid semantic chunking functionality
"""

import os
import sys
import django

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_backend.settings')
django.setup()

from api.document_processor import DocumentProcessor

def test_hybrid_chunking():
    """Test the hybrid semantic chunking functionality"""
    
    # Initialize document processor
    processor = DocumentProcessor()
    
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
    
    # Test hybrid chunking
    try:
        chunks = processor.hybrid_semantic_chunking(structured_text)
        
        print(f"\nGenerated {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Type: {chunk.get('type', 'unknown')}")
            print(f"Method: {chunk.get('chunk_method', 'unknown')}")
            print(f"Confidence: {chunk.get('confidence', 0.0):.2f}")
            print(f"Quality Score: {chunk.get('quality_score', 0.0):.2f}")
            print(f"Text: {chunk['text'][:100]}...")
            
            if 'metadata' in chunk:
                metadata = chunk['metadata']
                print(f"Medical Terms: {metadata.get('medical_terms', [])}")
                print(f"Sentence Count: {metadata.get('sentence_count', 0)}")
                print(f"Overlap Size: {metadata.get('overlap_size', 0)}")
    
    except Exception as e:
        print(f"Error in hybrid chunking: {e}")
    
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
    
    print("\n\n=== Testing Unstructured Medical Document ===")
    print(f"Original text length: {len(unstructured_text)} characters")
    
    try:
        chunks = processor.hybrid_semantic_chunking(unstructured_text)
        
        print(f"\nGenerated {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Type: {chunk.get('type', 'unknown')}")
            print(f"Method: {chunk.get('chunk_method', 'unknown')}")
            print(f"Confidence: {chunk.get('confidence', 0.0):.2f}")
            print(f"Quality Score: {chunk.get('quality_score', 0.0):.2f}")
            print(f"Text: {chunk['text'][:100]}...")
            
            if 'metadata' in chunk:
                metadata = chunk['metadata']
                print(f"Medical Terms: {metadata.get('medical_terms', [])}")
                print(f"Sentence Count: {metadata.get('sentence_count', 0)}")
    
    except Exception as e:
        print(f"Error in hybrid chunking: {e}")
    
    # Test document structure analysis
    print("\n\n=== Testing Document Structure Analysis ===")
    
    try:
        structure_analysis = processor.analyze_document_structure(structured_text)
        print(f"Structured Document Analysis:")
        print(f"  Score: {structure_analysis['score']:.2f}")
        print(f"  Is Structured: {structure_analysis['is_structured']}")
        print(f"  Medical Terms Count: {structure_analysis['medical_terms_count']}")
        
        structure_analysis = processor.analyze_document_structure(unstructured_text)
        print(f"\nUnstructured Document Analysis:")
        print(f"  Score: {structure_analysis['score']:.2f}")
        print(f"  Is Unstructured: {structure_analysis['is_unstructured']}")
        print(f"  Medical Terms Count: {structure_analysis['medical_terms_count']}")
    
    except Exception as e:
        print(f"Error in structure analysis: {e}")

if __name__ == "__main__":
    test_hybrid_chunking()
