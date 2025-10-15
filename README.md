# BERT Resume Classifier

A deep learning-powered tool for automatic resume screening and job role classification using BERT and NLP.

## Overview

This project implements an AI-powered resume screener that classifies resumes into job categories (e.g., Data Science, Web Development, HR, etc.) using a fine-tuned BERT model. It streamlines the recruitment process by automating resume sorting and candidate-job matching.

## Features

- Classifies resumes into 25+ job categories
- Built with Hugging Face Transformers and PyTorch
- High accuracy (98%+ on validation set)
- Ready for deployment as a Streamlit web app

## Dataset

- Source: [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- Preprocessed and labeled for multi-class classification

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository:**

git clone https://github.com/Exryz/resume-screening-bert.git

2. **Install dependencies:**

pip install -r requirements.txt

3. **(Optional) Create and activate a virtual environment:**

python -m venv venv

On Windows:
venv\Scripts\activate

On macOS/Linux:
source venv/bin/activate

### Training

To train the model on your own data:

Or use the provided Jupyter notebook for step-by-step training.

### Inference & Streamlit App

1. **Save the trained model:**

model.save_pretrained("resume_bert_model")
tokenizer.save_pretrained("resume_bert_model")

2. **Run the Streamlit app:**

streamlit run app.py

3. **Deploy to Streamlit Cloud:**
- Push your repo to GitHub.
- Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and deploy your app.

## Usage

- Paste or upload resume text in the app.
- The model predicts the most likely job category and shows class probabilities.

## Results

- **Validation Accuracy:** 98.45%
- See `notebook.ipynb` or `train.py` for detailed metrics.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- Hugging Face Transformers
- Streamlit

---
