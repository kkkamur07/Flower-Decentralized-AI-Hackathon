"""Streamlit app for Medical Image Classification using Federated Learning."""

import streamlit as st
import requests
import io
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
from datetime import datetime
import base64

# Configuration
API_BASE_URL = "http://localhost:8000"

# Dataset information for medical use cases
DATASET_INFO = {
    "pathmnist": {
        "title": "Tissue Pathology Analysis",
        "use_case": "Analyze Colon Tissue Samples",
        "description": "Identify tissue types and detect pathological changes in colon biopsies",
        "icon": "🔬",
        "color": "#FF6B6B",
        "clinical_purpose": "Histopathological diagnosis for colorectal conditions"
    },
    "retinamnist": {
        "title": "Retinal Health Screening",
        "use_case": "Screen Retinal OCT Images",
        "description": "Detect diabetic retinopathy, glaucoma, and other retinal disorders",
        "icon": "👁️",
        "color": "#4ECDC4",
        "clinical_purpose": "Early detection of retinal diseases"
    },
    "dermamnist": {
        "title": "Skin Lesion Evaluation",
        "use_case": "Evaluate Skin Lesions",
        "description": "Assess skin lesions for malignancy and classification",
        "icon": "🔍",
        "color": "#45B7D1",
        "clinical_purpose": "Skin cancer screening and dermatological diagnosis"
    },
    "bloodmnist": {
        "title": "Blood Cell Analysis",
        "use_case": "Analyze Blood Samples",
        "description": "Identify and count different blood cell types for hematological assessment",
        "icon": "🩸",
        "color": "#96CEB4",
        "clinical_purpose": "Complete blood count and differential analysis"
    }
}

def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def classify_with_summary(dataset: str, image_file) -> Dict[Any, Any]:
    """Send image to API for classification with medical summary."""
    try:
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        response = requests.post(
            f"{API_BASE_URL}/classify_with_summary/{dataset}",
            files=files,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Analysis Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to analysis service. Please contact IT support.")
        return None
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None

def generate_pdf_report(classification, medical_summary, dataset_info, filename):
    """Generate a medical report in HTML format for download."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
            .section {{ margin: 20px 0; }}
            .result {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; }}
            .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏥 Medical Image Analysis Report</h1>
            <p><strong>Report Generated:</strong> {timestamp}</p>
            <p><strong>Analysis Type:</strong> {dataset_info['title']}</p>
            <p><strong>Sample ID:</strong> {filename}</p>
        </div>
        
        <div class="section result">
            <h2>🎯 Primary Findings</h2>
            <p><strong>Classification:</strong> {classification['predicted_class']}</p>
            <p><strong>Confidence Level:</strong> {classification['confidence']:.2%}</p>
            <p><strong>Clinical Purpose:</strong> {dataset_info['clinical_purpose']}</p>
        </div>
        
        <div class="section">
            <h2>⚡ Clinical Assessment</h2>
            <p><strong>Confidence Level:</strong> {medical_summary['confidence_level']}</p>
            <p><strong>Urgency Level:</strong> {medical_summary['urgency']}</p>
            <p><strong>Recommended Actions:</strong> {medical_summary['next_steps']}</p>
        </div>
        
        <div class="section summary">
            <h2>🤖 AI Medical Interpretation</h2>
            <div>{medical_summary['summary']}</div>
        </div>
        
        <div class="section">
            <h2>📊 Classification Probabilities</h2>
            <table border="1" style="width:100%; border-collapse: collapse;">
                <tr><th>Classification</th><th>Probability</th></tr>
    """
    
    for class_name, prob in classification['class_probabilities'].items():
        html_content += f"<tr><td>{class_name.replace('_', ' ').title()}</td><td>{prob:.2%}</td></tr>"
    
    html_content += """
            </table>
        </div>
        
        <div class="section" style="border-top: 1px solid #ccc; padding-top: 20px; font-size: 12px; color: #666;">
            <p><strong>Medical Disclaimer:</strong> This AI-assisted analysis is for diagnostic support only and should be validated by qualified medical professionals. Clinical correlation is essential for accurate diagnosis and treatment planning.</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def display_results(result: Dict[Any, Any], dataset_info: Dict[str, str]):
    """Display analysis results with medical interpretation."""
    if not result or result.get("status") != "success":
        st.error("❌ Analysis failed or returned invalid results")
        return
    
    classification = result["classification"]
    medical_summary = result["medical_summary"]
    
    # Results header
    st.markdown("### 📋 Analysis Results")
    
    # Primary findings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🎯 Primary Classification")
        # Format the classification name
        formatted_class = classification['predicted_class'].replace('_', ' ').title()
        
        # Determine confidence level text and merge with classification
        if classification['confidence'] >= 0.8:
            confidence_text = "High Confidence"
            st.success(f"**{formatted_class}** ({confidence_text} - {classification['confidence']:.1%})")
        elif classification['confidence'] >= 0.6:
            confidence_text = "Moderate Confidence"
            st.warning(f"**{formatted_class}** ({confidence_text} - {classification['confidence']:.1%})")
        else:
            confidence_text = "Low Confidence"
            st.error(f"**{formatted_class}** ({confidence_text} - {classification['confidence']:.1%})")
    
    with col2:
        st.markdown("#### ℹ️ Sample Information")
        st.write(f"**Sample ID:** {classification['filename']}")
        st.write(f"**Analysis Type:** {dataset_info['title']}")
        st.write(f"**Total Classes:** {classification['num_classes']}")

    # VISUAL ANALYSIS SECTION
    st.markdown("### 📊 Visual Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = classification['confidence']*100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Analysis Confidence"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': dataset_info['color']},
                'steps': [
                    {'range': [0, 60], 'color': "#ffcccc"},
                    {'range': [60, 80], 'color': "#ffffcc"},
                    {'range': [80, 100], 'color': "#ccffcc"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Clinical assessment panel
        st.markdown("#### ⚡ Clinical Assessment")
        
        # Confidence assessment
        confidence_color = "🟢" if classification['confidence'] > 0.8 else "🟡" if classification['confidence'] > 0.6 else "🔴"
        st.markdown(f"**Reliability:** {confidence_color} {medical_summary['confidence_level']}")
        
        # Urgency assessment
        urgency_color = "🔴" if "High" in medical_summary['urgency'] else "🟡" if "Moderate" in medical_summary['urgency'] else "🟢"
        st.markdown(f"**Priority:** {urgency_color} {medical_summary['urgency']}")
        
        # Recommendations
        st.markdown("**Recommended Actions:**")
        st.info(medical_summary['next_steps'])
    
    # Probability distribution chart
    st.markdown("### 📈 Classification Probabilities")
    
    # Prepare and format data
    probs_df = pd.DataFrame([
        {"Classification": class_name.replace('_', ' ').title(), "Probability": prob}
        for class_name, prob in classification['class_probabilities'].items()
    ]).sort_values('Probability', ascending=True)
    
    # Horizontal bar chart
    fig_bar = px.bar(
        probs_df, 
        x='Probability', 
        y='Classification',
        orientation='h',
        color='Probability',
        color_continuous_scale='RdYlGn',
        title="Probability Distribution Across All Classifications"
    )
    fig_bar.update_layout(height=400, showlegend=False)
    fig_bar.update_traces(texttemplate='%{x:.1%}', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # MEDICAL INTERPRETATION SECTION
    st.markdown("---")
    st.markdown("### 🩺 Medical Interpretation")
    
    # Use expandable section for detailed summary
    with st.expander("📋 Detailed Clinical Analysis", expanded=True):
        st.markdown(medical_summary["summary"])
    
    # Add a note about validation
    st.info("💡 **Note:** This AI analysis should be reviewed and validated by a qualified medical professional before making clinical decisions.")

def create_dataset_tab(dataset_name: str, dataset_info: Dict[str, str]):
    """Create a medical analysis tab for specific dataset."""
    
    # Header section
    st.markdown(f"## {dataset_info['icon']} {dataset_info['use_case']}")
    st.markdown(f"**Clinical Purpose:** {dataset_info['clinical_purpose']}")
    st.markdown(f"*{dataset_info['description']}*")
    
    st.markdown("---")
    
    # File upload section
    st.markdown("### 📁 Upload Medical Image")
    
    uploaded_file = st.file_uploader(
        f"Select medical image for {dataset_info['title'].lower()}",
        type=['png', 'jpg', 'jpeg', 'tiff'],
        key=f"uploader_{dataset_name}",
        help=f"Upload a medical image for {dataset_info['title'].lower()} analysis"
    )
    
    if uploaded_file is not None:
        # Display section
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🖼️ Sample Preview")
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Sample: {uploaded_file.name}", use_container_width=True)
            
            # Analysis button
            if st.button(
                f"🚀 Start {dataset_info['title']} Analysis", 
                key=f"analyze_{dataset_name}", 
                type="primary",
                use_container_width=True
            ):
                with st.spinner(f"🔬 Processing {dataset_info['title'].lower()}..."):
                    result = classify_with_summary(dataset_name, uploaded_file)
                    if result:
                        # Store result in session state
                        st.session_state[f'result_{dataset_name}'] = result
                        st.rerun()
        
        with col2:
            st.markdown("### 🔬 Initiate Analysis")
            
            # Analysis information
            st.info(f"""
            **Analysis Type:** {dataset_info['title']}
            
            **Sample ID:** {uploaded_file.name}
            
            **Clinical Use:** {dataset_info['clinical_purpose']}
            """)
        
        # Check if we have results to display
        if f'result_{dataset_name}' in st.session_state:
            result = st.session_state[f'result_{dataset_name}']
            classification = result["classification"]
            medical_summary = result["medical_summary"]
            
            # Display all analysis results
            display_results(result, dataset_info)
            
            # DOWNLOAD BUTTON AT THE END
            st.markdown("---")
            st.markdown("### 📄 Export Analysis Report")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:  # Center the download button
                pdf_content = generate_pdf_report(classification, medical_summary, dataset_info, classification['filename'])
                st.download_button(
                    label="📄 Download Complete Medical Report",
                    data=pdf_content,
                    file_name=f"medical_analysis_{classification['filename']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    type="primary",
                    use_container_width=True
                )
            
            # Final note
            st.caption("💡 The complete analysis report includes all findings, clinical assessments, and AI interpretation for medical review.")

def main():
    """Main medical analysis application."""
    # Page configuration
    st.set_page_config(
        page_title="Medical Image Analysis Laboratory",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Application header
    st.title("🏥 Medical Image Analysis Laboratory")
    st.markdown("### *AI-Powered Diagnostic Support System*")
    
    # Sidebar - Laboratory information
    with st.sidebar:
        st.markdown("## 🔬 Laboratory Status")
        if check_api_health():
            st.success("🟢 Analysis System Online")
            st.caption("All diagnostic modules operational")
        else:
            st.error("🔴 Analysis System Offline")
            st.caption("Contact technical support")
        
        st.markdown("---")
        
        st.markdown("## 🧠 AI System Information")
        st.markdown("""
        **Architecture:** Deep Learning Neural Network
        
        **Training:** Federated Learning Protocol
        
        **Privacy:** HIPAA Compliant
        
        **Validation:** Multi-institutional dataset
        """)
        
        st.markdown("---")
        
        st.markdown("## 📋 Available Analyses")
        for dataset, info in DATASET_INFO.items():
            st.markdown(f"• {info['icon']} {info['title']}")
        
        st.markdown("---")
        st.caption("🔒 All patient data remains secure and private")
    
    # System availability check
    if not check_api_health():
        st.error("⚠️ **System Unavailable**: The AI analysis system is currently offline.")
        st.markdown("**Resolution Steps:**")
        st.code("1. Contact IT Support: ext. 2500\n2. Verify network connection\n3. Restart analysis service if authorized")
        st.stop()
    
    # Analysis tabs
    tabs = st.tabs([
        f"{info['icon']} {info['use_case']}" 
        for dataset, info in DATASET_INFO.items()
    ])
    
    # Create analysis tabs
    for tab, (dataset_name, dataset_info) in zip(tabs, DATASET_INFO.items()):
        with tab:
            create_dataset_tab(dataset_name, dataset_info)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        🏥 Medical Image Analysis Laboratory | 🤖 AI-Powered Diagnostics | 🔒 HIPAA Compliant System
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()