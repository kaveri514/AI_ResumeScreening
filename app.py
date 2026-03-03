import streamlit as st
import pandas as pd
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Resume ATS", layout="wide")

# Download stopwords
@st.cache_resource
def download_nltk():
    nltk.download('stopwords')

download_nltk()

# Load BERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

model = load_model()

# ------------------ SKILL DATABASE ------------------
skills_db = [
    "python", "java", "c++", "sql", "javascript",
    "machine learning", "deep learning", "nlp",
    "tensorflow", "pytorch", "scikit-learn",
    "react", "node", "flask", "django",
    "aws", "azure", "docker", "kubernetes",
    "pandas", "numpy", "power bi", "tableau",
    "communication", "teamwork", "problem solving"
]

# ------------------ FUNCTIONS ------------------

def extract_text(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)


def extract_skills(text):

    text = text.lower()
    found = []

    for skill in skills_db:
        # exact phrase match using word boundaries
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found.append(skill)

    return list(set(found))


def calculate_ats(job_skills, resume_skills):
    if not job_skills:
        return 0

    matched = set(job_skills).intersection(set(resume_skills))

    skill_score = (len(matched) / len(job_skills)) * 70
    coverage_score = (len(resume_skills) / len(skills_db)) * 30

    total_score = skill_score + coverage_score
    return round(min(total_score, 100), 2)


def suggest_keywords(job_text, resume_text):
    job_words = set(job_text.split())
    resume_words = set(resume_text.split())

    missing_keywords = []
    for word in job_words:
        if word not in resume_words and len(word) > 4:
            missing_keywords.append(word)

    return missing_keywords[:10]


def generate_resume_feedback(missing_skills, similarity):
    feedback = []

    if similarity < 0.5:
        feedback.append("Your resume does not strongly match the job description. Improve your summary aligned to this role.")

    if missing_skills:
        feedback.append(f"Add these technical skills: {', '.join(missing_skills)}")

    if not feedback:
        feedback.append("Resume is well aligned with the job description.")

    return feedback


def generate_pdf_report(filename, similarity, ats_score, missing_skills):
    pdf_path = f"{filename}_ATS_Report.pdf"
    doc = SimpleDocTemplate(pdf_path)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("AI Resume Screening Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Semantic Match: {similarity}%", styles['Normal']))
    elements.append(Paragraph(f"ATS Score: {ats_score}%", styles['Normal']))
    elements.append(Paragraph(f"Missing Skills: {', '.join(missing_skills)}", styles['Normal']))

    doc.build(elements)
    return pdf_path

# ------------------ UI ------------------

st.title("📄 AI Resume Screening & ATS Analysis")

job_description = st.text_area("Paste Job Description Here")
uploaded_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

# ------------------ ANALYSIS ------------------

if st.button("Analyze Resumes"):

    if not job_description or not uploaded_files:
        st.warning("Please provide job description and upload resumes.")
    else:

        with st.spinner("Running AI Analysis..."):

            job_clean = preprocess(job_description)
            job_skills = extract_skills(job_description)
            job_embedding = model.encode(job_clean)

            for file in uploaded_files:

                resume_text = extract_text(file)
                resume_clean = preprocess(resume_text)
                resume_skills = extract_skills(resume_clean)
                resume_embedding = model.encode(resume_clean)

                similarity = cosine_similarity(
                    [job_embedding],
                    [resume_embedding]
                )[0][0]

                missing_skills = list(set(job_skills) - set(resume_skills))
                ats_score = calculate_ats(job_skills, resume_skills)
                keyword_suggestions = suggest_keywords(job_clean, resume_clean)
                feedback = generate_resume_feedback(missing_skills, similarity)
                
                st.markdown("-----")
                st.subheader(f"📌 {file.name}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Semantic Match", f"{round(similarity*100,2)}%")
                col2.metric("ATS Score", f"{ats_score}%")
                col3.metric("Skill Coverage", f"{len(resume_skills)}/{len(job_skills)}")

                st.progress(ats_score / 100)

                st.write("### 🎯 Job Skills Detected")
                st.info(", ".join(job_skills) if job_skills else "None")

                st.write("### ✅ Resume Skills Detected")
                st.success(", ".join(resume_skills) if resume_skills else "None")

                st.write("### ❌ Missing Skills")
                st.error(", ".join(missing_skills) if missing_skills else "None")

                st.write("### 🔑 Suggested Keywords From JD")
                st.warning(", ".join(keyword_suggestions) if keyword_suggestions else "None")

                st.write("### 🤖 Resume Suggestions")
                for f in feedback:
                    st.info(f)

                pdf_path = generate_pdf_report(
                    file.name,
                    round(similarity * 100, 2),
                    ats_score,
                    missing_skills
                )

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📥 Download ATS Report",
                        f,
                        file_name=pdf_path
                    )