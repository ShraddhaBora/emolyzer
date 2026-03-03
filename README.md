# Emolyzer

Emolyzer is an emotion analysis platform that uses a multi-model NLP pipeline to classify text into core emotional states. I built this with a modern React frontend (using TypeScript and React Query) and a modular FastAPI backend (utilizing SQLAlchemy and rate limiting). The goal is to provide a clean, research-grade interface for real-time sentiment extraction.

## Features

- Multi-Model Analysis: Evaluates text using Logistic Regression, Naive Bayes, and Support Vector Machines with Platt scaling.
- Deep Linguistic Pipeline: Includes conjunction-aware negation marking and high-intensity signal boosting.
- Persistent Data Tracking: A local SQLite database automatically tracks dataset metadata and analysis over time.
- Academic Aesthetic: Features a minimalist UI with a soft pastel palette and smooth, staggered transitions provided natively by Framer Motion.
- Developer Experience: Includes fully typed APIs, Dockerized environment configuration, and automated code formatting with Prettier and Ruff.

## Project Structure

- frontend/: React application using Vite, TypeScript, React Query, Framer Motion, and Recharts.
- routers/: Modularized FastAPI routers for decoupled predictions, datasets, and model endpoints.
- src/: Core logic for preprocessing, SQLite connection mapping (database.py, models_db.py), and model training.
- models/: Where the persisted champion models and evaluation metadata are saved per version.

## Running the Project

The easiest way to get the stack running is by using Docker Compose:

```bash
docker compose up --build
```

This starts both the backend API on port 8000 and the React frontend on port 5173. Hot-reloading is enabled for an optimized development workflow.

### Local Installation without Docker

If you prefer to run it locally without Docker, follow these steps:

#### 1. Start the Backend

Create a virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
pip install slowapi sqlalchemy pydantic-settings ruff

uvicorn api:app --reload --port 8000
```

#### 2. Start the Frontend

Navigate to the frontend directory, install the packages, and start the development server:

```bash
cd frontend
npm install
npm run dev
```

## Formatting and Linting

- Frontend: Run `npm run format` (uses Prettier).
- Backend: Run `python -m ruff format . && python -m ruff check .`

## Methodology

The system leverages a stratified 5-fold cross-validation approach on a combined corpus of roughly 470K samples from Reddit, Twitter, and DAIR-AI datasets. It maps these emotional signals to 7 core classes: Sadness, Joy, Love, Anger, Fear, Surprise, and Neutral.

## Running Tests

To execute the unit tests, use pytest:

```bash
pytest tests/
```
