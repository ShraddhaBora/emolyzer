# Emolyzer

Emolyzer is an emotion analysis application built with Streamlit. It uses machine learning models to analyze and predict emotions from text input.

## Features

* **Interactive Interface:** A beautiful, pastel-themed Streamlit UI.
* **Text Analysis:** Input text to get emotion predictions.
* **Machine Learning Models:** Leverages trained NLP models to classify text.

## Project Structure

* `src/`: Contains core application code (data utilities, model pipelines, preprocessing).
* `scripts/`: Scripts for building and managing the dataset.
* `tests/`: Unit tests for ensuring code reliability.
* `app.py`: The main entry point for the Streamlit application.
* `.streamlit/`: Configuration for the Streamlit app.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ShraddhaBora/emolyzer.git
   cd emolyzer
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To start the Streamlit app, run the following command in your terminal:

```bash
streamlit run app.py
```

## Running Tests

To run the unit tests, use pytest:

```bash
pytest tests/
```
