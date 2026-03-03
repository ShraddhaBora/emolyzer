# Emolyzer

Emolyzer is an advanced emotion analysis platform that leverages a multi-model NLP pipeline to classify text into core emotional states. Built with a React frontend and a FastAPI backend, it provides a research-grade interface for real-time sentiment extraction.

##  Features

*   **Multi-Model Analysis:** Evaluates text using Logistic Regression, Naive Bayes, and SVM with Platt scaling.
*   **Deep Linguistic Pipeline:** Conjunction-aware negation marking and high-intensity signal boosting.
*   **Research Aesthetic:** Minimalist, academic-style UI with a soft pastel palette and smooth staggered transitions.
*   **Real-time Metrics:** Stratified cross-validation results and confidence calibration shown in live charts.

##  Project Structure

*   `frontend/`: React application using Vite, Framer Motion, and Recharts.
*   `api.py`: FastAPI backend that exposes the NLP pipeline endpoints.
*   `src/`: Core logic for preprocessing, model training, and data handling.
*   `models/`: Persisted champion model and evaluation metadata.

##  Running the Project

### 1. Start the Backend
```bash
python -m uvicorn api:app --reload --port 8000
```

### 2. Start the Frontend
```bash
cd frontend
npm run dev
```

##  Methodology

The system uses a stratified 5-fold cross-validation approach on a corpus of ~470K samples (Reddit, Twitter, and DAIR-AI datasets). It maps emotional signals to 7 core classes: Sadness, Joy, Love, Anger, Fear, Surprise, and Neutral.

##  Running Tests

To run the unit tests, use pytest:
```bash
pytest tests/
```
