import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Hugging Face Hub repo path
repo_id = "Exeryze/resume-screening-bert"

# Cache model and tokenizer loading for performance
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(repo_id)
    tokenizer = BertTokenizer.from_pretrained(repo_id)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Replace with your actual label names
label_names = [
    "Data Science", "HR", "Advocate", "Arts", "Web Designing", "Mechanical Engineer",
    "Sales", "Health and fitness", "Civil Engineer", "Java Developer", "Business Analyst",
    "SAP Developer", "Automation Testing", "Electrical Engineering", "Operations Manager",
    "Python Developer", "DevOps Engineer", "Network Security Engineer", "PMO", "Database",
    "Hadoop", "ETL Developer", "DotNet Developer", "Blockchain", "Testing"
]

st.title("Resume Category Classifier")
resume_text = st.text_area("Paste resume text here:")

if st.button("Classify"):
    if resume_text.strip():
        inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            st.write(f"**Predicted Category:** {label_names[pred_label]}")
            st.write("**Probabilities:**")
            for i, prob in enumerate(probs[0]):
                st.write(f"{label_names[i]}: {prob.item():.4f}")
    else:
        st.warning("Please enter some resume text.")
