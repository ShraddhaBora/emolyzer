# Emolyzer

Emolyzer is an advanced emotion analysis platform utilizing a multi-model NLP pipeline to classify text into core emotional states. Engineered with a scalable React frontend and a modular FastAPI backend, the platform provides a robust, research-grade interface designed for real-time sentiment extraction.

## Core Features

- **Multi-Model Analysis**: Evaluates text using Logistic Regression, Naive Bayes, and Support Vector Machines equipped with Platt scaling for probability calibration.
- **Deep Linguistic Pipeline**: Incorporates conjunction-aware negation marking and high-intensity signal boosting for accurate contextual understanding.
- **Persistent Data Tracking**: Integrates a local SQLite database with SQLAlchemy ORM to automatically track dataset metadata and model performance over time.
- **Academic Aesthetic**: Features a clean, minimalist UI utilizing a soft pastel color palette, with smooth, staggered transitions provided natively by Framer Motion.
- **Enhanced Developer Experience**: Built with fully defined TypeScript interfaces for API validation, Dockerized environment orchestration, and automated code formatting with Prettier and Ruff.

## System Architecture

The repository is structured to separate concerns, allowing for independent scaling and maintenance of the frontend presentation and backend machine learning logic:

- **`frontend/`**: The client-side React application built with Vite, TypeScript, React Query (for efficient server-state management), Framer Motion, and Recharts.
- **`routers/`**: The backend API layer, consisting of modularized FastAPI routers decoupling the prediction, dataset management, and model handling endpoints.
- **`src/`**: The core Python package containing logic for NLP preprocessing, database connection mapping (`database.py`, `models_db.py`), and machine learning model training pipelines.
- **`models/`**: The storage directory where persisted champion models and their associated evaluation metadata are versioned and saved.

## Getting Started

### Quick Start with Docker

The most straightforward method to run the complete stack is through Docker Compose:

```bash
docker compose up --build
```

This command provisions and starts both the backend API on port `8000` and the React frontend on port `5173`. Hot-reloading is configured by default to ensure an optimized development workflow.

### Local Installation without Docker

If you prefer to run the components independently on your local machine, follow the steps below:

#### 1. Start the Backend API

First, create a virtual environment and install the required Python dependencies:

```bash
pip install -r requirements.txt
pip install slowapi sqlalchemy pydantic-settings ruff

python -m uvicorn api:app --reload --port 8000
```

#### 2. Start the Frontend Application

Next, navigate to the frontend directory, install the Node packages, and start the Vite development server:

```bash
cd frontend
npm install
npm run dev
```

## Development and Contribution

To maintain code quality across the repository, the project relies on specific formatting and linting tools.

- **Frontend**: Execute `npm run format` to apply Prettier formatting.
- **Backend**: Execute `python -m ruff format . && python -m ruff check .` to format and check the Python codebase.

## Research Methodology

The classification system leverages a stratified 5-fold cross-validation approach on a combined text corpus consisting of approximately 470,000 samples gathered from Reddit, Twitter, and DAIR-AI datasets. It maps detected emotional signals to seven core classes: Sadness, Joy, Love, Anger, Fear, Surprise, and Neutral.

## Testing

To verify the integrity of the data processing and machine learning pipelines, run the included test suite with pytest:

```bash
pytest tests/
```
