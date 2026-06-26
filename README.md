# HARAYA

## About the project
HARAYA is a lightweight web application that analyzes a conversation and estimates whether it looks safe, potentially harassing, or clearly harassing. It combines a machine-learning classifier with behavioral signals such as message frequency, repetition, and sentiment.

## The problem it solves
The project helps users spot concerning interaction patterns early. Instead of manually reviewing a chat thread, HARAYA provides a quick risk assessment, explanation, legal references, and suggested next steps.

## Technologies used
- Frontend: HTML, CSS, and JavaScript
- Backend: FastAPI (Python)
- ML/NLP: PyTorch, Transformers, scikit-learn, NumPy, pandas
- Data handling and preprocessing: Python scripts for training and evaluation

## Contributions
- Built the full conversational web UI for entering messages and viewing analysis results
- Implemented the backend API for sending conversations to the classifier
- Designed the prediction pipeline using both model-based and rule-based signals
- Added explanation output such as reasons, legal basis, and recommendations
- Prepared training and evaluation scripts for the model workflow

## How to run
1. Open a terminal in the project root.
2. Install Python dependencies:
   ```bash
   pip install -r python_backend/requirements.txt
   ```
3. Start the backend API:
   ```bash
   uvicorn python_backend.haraya.app:app --reload --host 0.0.0.0 --port 8000
   ```
4. In another terminal, serve the frontend:
   ```bash
   python -m http.server 8080
   ```
5. Open http://localhost:8080/html/ in your browser.

> If trained model weights are not available yet, the app can still produce a fallback response using deterministic rules.

## Interface
The app presents a simple chat-style interface where users can:
- add messages from two speakers
- build a conversation log
- run an analysis
- view a classification result with confidence, score, recommended actions, and legal context
