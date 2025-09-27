"""Test script for LLM service."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medapp.service.llm import llm_service

def test_llm_service():
    """Test the LLM service with sample data."""
    
    # Sample classification result
    sample_result = {
        "dataset": "dermamnist",
        "description": "Dermatology classification",
        "predicted_class": "melanoma",
        "confidence": 0.85,
        "class_probabilities": {
            "melanoma": 0.85,
            "melanocytic_nevi": 0.10,
            "basal_cell_carcinoma": 0.05
        },
        "filename": "test_sample.jpg"
    }
    
    print("🔬 Testing LLM Service...")
    print(f"Service Available: {llm_service.is_available}")
    print(f"Model: {llm_service.model_name}")
    print(f"Base URL: {llm_service.base_url}")
    
    # Test medical summary generation
    print("\n📋 Generating Medical Summary...")
    summary = llm_service.generate_medical_summary(sample_result)
    
    print("\n✅ Results:")
    print(f"Summary: {summary['summary'][:200]}...")
    print(f"Confidence Level: {summary['confidence_level']}")
    print(f"Urgency: {summary['urgency']}")
    print(f"Next Steps: {summary['next_steps']}")

if __name__ == "__main__":
    test_llm_service()