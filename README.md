# 🏥 MedX-Lab: AI-Powered Diagnostic Laboratory  

---
## 🎥 Demo  
👉 *(Add video or screenshots here)*  

---
MedX-Lab is a **next-generation diagnostic platform** that allows you to upload medical samples (images, scans, etc.) and instantly receive an AI-generated diagnosis with follow-up medical recommendations.  

Traditionally, diagnosis and report preparation can take **days** — MedX-Lab reduces this to **minutes**.  
Doctors can quickly vet AI-prepared reports, accelerating clinical decision-making and patient care.  

⚡ Built with **Federated Learning** with **flower framework**:  
- No patient data sharing issues (privacy preserved)  
- AI is trained across multiple institutions without centralizing sensitive data  
- Runs **locally**, ensuring compliance and security  

---

| Step                         | Human (Traditional) | AI (MedX-Lab) |
|------------------------------|----------------------|---------------|
| Sample upload & pre-check    | 2–4 hours           | < 1 minute    |
| Initial classification       | 1–2 days            | < 5 minutes   |
| Report preparation           | 1 day               | < 2 minutes   |
| Doctor review & validation   | 1–2 hours           | ~1 hour (vetting only) |
| **Total Time**               | **2–4 days**        | **~10 minutes + doctor vetting** |

---

```mermaid
flowchart LR 
  A[Upload the image] --> B[Get the diagnosis]
  B --> C[Get the report]
```

---

## ✨ Key Features
- 📂 **Upload Medical Images** (retina, skin, tissue, blood, etc.)  
- 🤖 **Classification & Diagnosis** using trained deep learning models  
- 📝 **Instant Report Generation** powered by a **locally hosted LLM**  
- 🔒 **Privacy First**: Data never leaves your local system  
- ⚡ **Federated Learning** ensures robust, secure, and distributed training  

---

## 📊 Supported Analyses
- 🔬 **Tissue Pathology Analysis** (Colon biopsies)  
- 👁️ **Retinal Health Screening** (Diabetic retinopathy, glaucoma, etc.)  
- 🔍 **Skin Lesion Evaluation** (Cancerous vs non-cancerous lesions)  
- 🩸 **Blood Cell Analysis** (Blood count and differential analysis)  

---

## 🛠️ Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/medx-lab.git
cd medx-lab
```

### Install the dependencies
```bash
pip install -r requirements.txt
uv pip install . # to use pyproject.toml
```

### 3️⃣ Start Backend (API)

```bash
uvicorn main:app --reload
```

### 4️⃣ Start Frontend (Streamlit)

```bash
streamlit run app.py
```

---

## 📄 Reports

- AI generates **HTML/PDF medical reports** with:
    
    - Primary findings & classification confidence
        
    - Clinical assessment (urgency, recommendations)
        
    - AI medical interpretation
        
    - Classification probabilities
        

---

## ⚠️ Disclaimer

This tool is for **diagnostic support only**.  
All AI outputs **must be validated** by qualified medical professionals before clinical use.

---

## 📌 Tech Stack

- **Frontend**: Streamlit
    
- **Backend**: FastAPI + Uvicorn
    
- **AI Models**: Federated Learning Neural Networks
    
- **Visualization**: Plotly, Pandas
    

---

## 🔒 Privacy & Compliance

- Fully **HIPAA-compliant**
    
- Runs **locally** (no cloud dependency)
    
- Federated learning ensures **no raw data exchange**
    

---

## 🚀 Future Roadmap

- 🔗 Integration with hospital EMR systems
- 🧪 Multi-modal diagnostics (text + imaging)  
- 📱 Mobile-first deployment