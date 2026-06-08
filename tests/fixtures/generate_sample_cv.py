"""One-off script: generates a synthetic ML Engineer junior CV as a real-text PDF."""

from fpdf import FPDF

CV_TEXT = """Alex Martin
ML Engineer Junior - Paris, France
alex.martin@example.com | +33 6 12 34 56 78 | linkedin.com/in/alexmartin

SUMMARY
Junior Machine Learning Engineer with 2 years of experience building and deploying
NLP and computer vision models. Passionate about MLOps practices and writing clean,
production-ready Python code. Currently completing a Machine Learning Engineer
certification at DataScientest.

EXPERIENCE
ML Engineer - TechNova (Paris)                                   2023 - Present
- Built and deployed a text classification pipeline using Python, PyTorch and
  Hugging Face Transformers, improving customer ticket routing accuracy by 18%.
- Designed CI/CD pipelines with GitHub Actions and Docker to automate model
  retraining and deployment on AWS.
- Collaborated with data scientists to track experiments using MLflow.

Data Analyst Intern - DataCorp (Lyon)                            2022 - 2023
- Used pandas, numpy and scikit-learn to analyze customer churn data.
- Built dashboards in Power BI and automated reporting with SQL and Python.

EDUCATION
Machine Learning Engineer Certification - DataScientest            2024 - 2025
Master's Degree in Applied Mathematics - Université Lyon 1         2020 - 2022

SKILLS
Python, SQL, Bash, Docker, Kubernetes, AWS, Git, FastAPI, Flask, PyTorch,
TensorFlow, scikit-learn, pandas, numpy, Hugging Face, NLP, MLflow, CI/CD,
GitHub Actions, Linux

PROJECTS
ATS CV Scorer - personal project
- End-to-end pipeline: PDF parsing, spaCy NLP, semantic scoring with
  sentence-transformers, and AI feedback generation via the Claude API,
  exposed through FastAPI and a Gradio interface.

Image Classifier for Plant Diseases - DataScientest capstone
- Trained a convolutional neural network with TensorFlow/Keras reaching 94%
  accuracy; deployed as a Streamlit app with Docker.

CERTIFICATIONS
AWS Certified Cloud Practitioner - Amazon Web Services (2024)
Machine Learning Specialization - DeepLearning.AI / Coursera (2023)

LANGUAGES
French (native), English (fluent, C1), Spanish (intermediate, B1)
"""


def main() -> None:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    for line in CV_TEXT.split("\n"):
        pdf.set_x(pdf.l_margin)
        if not line.strip():
            pdf.ln(4)
        elif line.isupper():
            pdf.ln(2)
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.write(7, line)
            pdf.ln(7)
            pdf.set_font("Helvetica", size=11)
        else:
            pdf.write(6, line)
            pdf.ln(6)

    pdf.output("tests/fixtures/sample_cv_ml_engineer_junior.pdf")
    print("PDF généré : tests/fixtures/sample_cv_ml_engineer_junior.pdf")


if __name__ == "__main__":
    main()
