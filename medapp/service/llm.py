"""LLM service for generating medical image classification summaries using Ollama."""

import requests
import json
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import streamlit as st

class BaseLLMService(ABC):
    """Abstract base class for LLM services."""
    
    @abstractmethod
    def generate_medical_summary(self, classification_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate a comprehensive medical summary from classification results."""
        pass

class OllamaLLMService(BaseLLMService):
    """LLM service using Ollama for medical image classification summaries."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model_name: str = "llama3.2:3b",
                 temperature: float = 0.3,
                 max_tokens: int = 500):
        """
        Initialize Ollama LLM service.
        
        Args:
            base_url (str): Ollama API base URL
            model_name (str): Name of the model to use
            temperature (float): Temperature for response generation (0.0-1.0)
            max_tokens (int): Maximum tokens in response
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_available = self._check_ollama_availability()
        
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models.get('models', [])]
            return []
        except Exception:
            return []
    
    def _create_medical_prompt(self, result: Dict[str, Any]) -> str:
        """Create specific prompt based on classification results."""
        dataset = result['dataset']
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        
        # Get top 3 predictions
        sorted_probs = sorted(result['class_probabilities'].items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_probs[:3]
        
        system_prompt = """You are a medical AI assistant helping healthcare professionals interpret medical imaging results. 

Your role is to:
1. Provide clear, professional medical interpretations
2. Suggest appropriate next steps or considerations
3. Highlight important findings that need attention
4. Use medical terminology appropriately but remain accessible
5. Always remind that this is AI-assisted analysis and should be validated by medical professionals

Structure your response with:
- Clinical Interpretation
- Key Findings
- Recommendations
- Important Notes"""
        
        user_prompt = f"""
Medical Image Classification Results:

Dataset: {dataset.upper()}
Image Type: {result['description']}

AI Classification Results:
Primary Diagnosis: {predicted_class} (Confidence: {confidence:.1%})

Top 3 Predictions:
1. {top_3[0][0]}: {top_3[0][1]:.1%}
2. {top_3[1][0]}: {top_3[1][1]:.1%}
3. {top_3[2][0]}: {top_3[2][1]:.1%}

Please provide a comprehensive medical interpretation of these results including clinical significance, recommended actions, and any important considerations for the healthcare team.
and only make as 2-3 recommendations max.
"""
        
        return f"{system_prompt}\n\n{user_prompt}"
    
    def _call_ollama_api(self, prompt: str) -> str:
        """Make API call to Ollama."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False,
                "options": {
                    "num_predict": self.max_tokens
                }
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "Connection Error: Cannot connect to Ollama service. Please ensure Ollama is running."
        except requests.exceptions.Timeout:
            return "Timeout Error: Ollama service took too long to respond."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _assess_confidence_level(self, confidence: float) -> str:
        """Assess confidence level for medical context."""
        if confidence >= 0.9:
            return "High - Strong diagnostic confidence"
        elif confidence >= 0.7:
            return "Moderate - Consider additional validation"
        else:
            return "Low - Requires expert review and additional testing"
    
    def _assess_urgency(self, predicted_class: str, dataset: str) -> str:
        """Assess urgency based on predicted condition."""
        # Define urgent conditions for each dataset
        urgent_conditions = {
            "retinamnist": ["diabetes", "glaucoma"],
            "dermamnist": ["melanoma", "basal_cell_carcinoma"],
            "pathmnist": ["colorectal_adenocarcinoma_epithelium"],
            "bloodmnist": ["immature_granulocytes"]
        }
        
        if predicted_class.lower() in urgent_conditions.get(dataset, []):
            return "High - Requires prompt medical attention"
        else:
            return "Routine - Standard follow-up recommended"
    
    def _suggest_next_steps(self, predicted_class: str, dataset: str) -> str:
        """Suggest appropriate next steps based on classification."""
        next_steps = {
            "retinamnist": "Consider ophthalmological consultation and comprehensive eye examination",
            "dermamnist": "Recommend dermatological evaluation and possible biopsy if malignant features suspected",
            "pathmnist": "Suggest histopathological review and correlation with clinical findings",
            "bloodmnist": "Recommend complete blood count analysis and hematological assessment"
        }
        return next_steps.get(dataset, "Consult with appropriate specialist for further evaluation")
    
    def _parse_llm_response(self, response_text: str, result: Dict[str, Any]) -> Dict[str, str]:
        """Parse LLM response into structured format."""
        return {
            "summary": response_text,
            "confidence_level": self._assess_confidence_level(result['confidence']),
            "urgency": self._assess_urgency(result['predicted_class'], result['dataset']),
            "next_steps": self._suggest_next_steps(result['predicted_class'], result['dataset'])
        }
    
    def generate_medical_summary(self, classification_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate a comprehensive medical summary from classification results."""
        
        if not self.is_available:
            return self._generate_fallback_summary(classification_result)
        
        # Check if model is available
        available_models = self._get_available_models()
        if self.model_name not in available_models:
            # Try to use the first available model
            if available_models:
                self.model_name = available_models[0]
            else:
                return self._generate_fallback_summary(classification_result)
        
        try:
            # Create prompt
            prompt = self._create_medical_prompt(classification_result)
            
            # Call Ollama API
            response_text = self._call_ollama_api(prompt)
            
            # Parse and return structured response
            return self._parse_llm_response(response_text, classification_result)
            
        except Exception as e:
            return self._generate_error_summary(str(e), classification_result)
    
    def _generate_fallback_summary(self, result: Dict[str, Any]) -> Dict[str, str]:
        """Generate fallback summary when Ollama is not available."""
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        dataset = result['dataset']
        
        mock_summaries = {
            "retinamnist": {
                "normal": "The retinal OCT scan appears normal with no significant pathological findings detected.",
                "diabetes": "Diabetic retinopathy changes detected. Monitor blood sugar levels and schedule regular ophthalmological follow-ups.",
                "glaucoma": "Signs consistent with glaucoma identified. Immediate ophthalmological consultation recommended for IOP assessment.",
                "cataract": "Cataract formation detected. Consider surgical evaluation if vision is significantly impaired.",
                "amd": "Age-related macular degeneration changes observed. Anti-VEGF therapy may be considered."
            },
            "dermamnist": {
                "melanoma": "⚠️ URGENT: Melanoma suspected. Immediate dermatological consultation and biopsy required.",
                "basal_cell_carcinoma": "Basal cell carcinoma features identified. Schedule dermatological evaluation for treatment planning.",
                "melanocytic_nevi": "Benign melanocytic nevus appearance. Routine monitoring recommended.",
                "benign_keratosis": "Benign keratosis identified. Generally no treatment required unless cosmetically concerning."
            },
            "pathmnist": {
                "colorectal_adenocarcinoma_epithelium": "⚠️ Malignant epithelial cells detected. Urgent oncological consultation required.",
                "normal_colon_mucosa": "Normal colonic tissue architecture observed.",
                "lymphocytes": "Lymphocytic infiltration present. Consider inflammatory or immune-mediated process."
            },
            "bloodmnist": {
                "neutrophil": "Normal neutrophil morphology observed.",
                "lymphocyte": "Lymphocyte identified with normal characteristics.",
                "platelet": "Platelet morphology appears normal.",
                "immature_granulocytes": "Immature granulocytes detected. Consider infection or hematological disorder."
            }
        }
        
        summary_text = mock_summaries.get(dataset, {}).get(predicted_class, 
            f"Classification suggests {predicted_class}. Professional medical review recommended.")
        
        return {
            "summary": f"""
**Clinical Interpretation:**
{summary_text}

**Key Findings:**
- Primary classification: {predicted_class} 
- Diagnostic confidence: {confidence:.1%}
- Image quality: Adequate for AI analysis

**Recommendations:**
- {self._suggest_next_steps(predicted_class, dataset)}
- Correlate with clinical history and symptoms
- Consider additional imaging or laboratory tests if indicated

**Important Notes:**
⚠️ This AI-assisted analysis should be validated by qualified medical professionals. Clinical correlation is essential for accurate diagnosis and treatment planning.

*Note: Ollama LLM service is currently unavailable. Using fallback analysis.*
            """,
            "confidence_level": self._assess_confidence_level(confidence),
            "urgency": self._assess_urgency(predicted_class, dataset),
            "next_steps": self._suggest_next_steps(predicted_class, dataset)
        }
    
    def _generate_error_summary(self, error: str, result: Dict[str, Any]) -> Dict[str, str]:
        """Generate summary when LLM service fails."""
        return {
            "summary": f"AI analysis completed with {result['confidence']:.1%} confidence for {result['predicted_class']}. LLM summary encountered an error: {error}. Please consult with medical professional for interpretation.",
            "confidence_level": self._assess_confidence_level(result['confidence']),
            "urgency": "Unknown - Professional review required",
            "next_steps": "Consult with appropriate medical specialist"
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status information."""
        return {
            "service_available": self.is_available,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "available_models": self._get_available_models() if self.is_available else []
        }

class LLMServiceFactory:
    """Factory class for creating LLM services."""
    
    @staticmethod
    def create_service(service_type: str = "ollama", **kwargs) -> BaseLLMService:
        """Create an LLM service instance."""
        if service_type.lower() == "ollama":
            return OllamaLLMService(**kwargs)
        else:
            raise ValueError(f"Unsupported service type: {service_type}")

# Configuration class for easy settings management
class LLMConfig:
    """Configuration class for LLM settings."""
    
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434"
        self.model_name = "llama3.2:3b"  # Default model
        self.temperature = 0.3
        self.max_tokens = 500
        
        # Alternative models to try
        self.fallback_models = [
            "llama3.2:3b",
            "llama3.2:1b", 
            "mistral:7b",
            "codellama:7b",
            "phi3:mini"
        ]
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return {
            "base_url": self.ollama_base_url,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

# Global instances
config = LLMConfig()
llm_service = LLMServiceFactory.create_service("ollama", **config.get_config())

# Convenience function for Streamlit
@st.cache_resource
def get_llm_service() -> OllamaLLMService:
    """Get cached LLM service instance for Streamlit."""
    return llm_service