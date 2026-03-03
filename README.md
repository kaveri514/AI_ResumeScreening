# 🚀 AI Resume ATS Analyzer

<p align="center">
  <b>An AI-powered Resume Screening System using NLP & Transformer Embeddings</b>
</p>

<p align="center">
  <a href="[(https://airesumescreening-vzsgugzjen6qghfxdfievz.streamlit.app/)]" target="_blank">
    🌐 Live Demo
  </a>
</p>

---

## 📌 Project Overview

Modern recruitment systems use **Applicant Tracking Systems (ATS)** to filter resumes before shortlisting candidates.

This project simulates a real-world ATS by:

- Extracting text from resumes (PDF)
- Comparing resumes with job descriptions
- Calculating semantic similarity scores
- Identifying missing skills
- Providing improvement suggestions
- Generating downloadable ATS reports



## ✨ Key Features

✔ Upload multiple resumes (PDF)  
✔ Paste Job Description  
✔ Skill extraction from Resume & JD  
✔ ATS Match Score (Cosine Similarity)  
✔ Missing skill detection  
✔ Keyword suggestions  
✔ Resume improvement feedback  
✔ Downloadable PDF report  

---

## 🛠 Tech Stack

- **Python**
- **Streamlit** (Frontend + Backend UI)
- **Sentence Transformers (MiniLM Model)** – Semantic embeddings
- **Scikit-learn** – Cosine similarity
- **NLTK** – Text preprocessing
- **PyPDF2** – Resume text extraction
- **ReportLab** – PDF report generation
- **Matplotlib** – Score visualization

---

## 🧠 How It Works

1. User uploads resume(s) in PDF format.
2. User enters job description.
3. Text is cleaned and preprocessed.
4. Sentence embeddings are generated using a transformer model.
5. Cosine similarity calculates ATS score.
6. Skill gap analysis is performed.
7. System provides:
   - Match percentage
   - Missing skills
   - Suggested keywords
   - Improvement feedback
8. User can download a structured ATS report in PDF format.

---

## 📊 Real-World Application

This project demonstrates:

- NLP-based semantic matching
- AI-driven recruitment automation
- Resume optimization strategies
- Practical use of transformer models

It simulates how modern recruitment platforms screen resumes at scale.

---

## 🏗 Installation (Run Locally)

```bash
git clone https://github.com/yourusername/AI_ResumeScreening.git
cd AI_ResumeScreening
pip install -r requirements.txt
streamlit run app.py

