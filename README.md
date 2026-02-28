# Fake Medicine Detector (MedGuard AI)

An AI-powered web application that detects counterfeit medicines by analyzing packaging images using image recognition.

## Features
- **AI Recognition**: Uses a pre-trained Deep Learning model (MobileNetV2) to identify medicine packaging.
- **Webcam Support**: Capture images directly from your webcam.
- **Modern UI**: Clean, responsive, and healthcare-themed design built with Vanilla CSS and JS.
- **Instant Result**: Real-time analysis with confidence scores and detailed reports.

## Tech Stack
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Lucide Icons.
- **Backend**: Python 3.x, FastAPI, Uvicorn.
- **AI/ML**: TensorFlow, Keras, Pillow, NumPy.

## Installation

### Prerequisites
- Python 3.8+
- Node.js (Optional, for serving frontend)

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the API server:
   ```bash
   python app.py
   ```

### Frontend Setup
1. Open `frontend/index.html` in any modern web browser.
2. Ensure the backend is running at `http://localhost:8000`.

## Project Structure
```text
fake-medicine-detector/
├── streamlit_app.py     # Alternative Streamlit Web App
├── backend/
│   ├── app.py           # FastAPI server & AI logic
│   ├── train.py         # Model training script
│   ├── requirements.txt # Python dependencies
│   └── dataset/         # Training dataset (if applicable)
├── frontend/
│   ├── index.html       # Landing Page & Scanner UI
│   ├── styles/
│   │   └── main.css     # Premium Styles
│   ├── js/
│   │   └── app.js       # App Logic & API Integration
│   └── assets/          # Images & Icons
├── models/
│   └── medicine_classifier.h5 # Pre-trained custom model (if committed)
├── .gitignore           # Git ignore file
└── README.md
```

## Running the Streamlit App (Alternative)
If you prefer to run the standalone Streamlit interface instead of the HTML/FastAPI stack:
1. Install dependencies from the `backend/` directory (`pip install -r backend/requirements.txt`)
2. Run from the root directory:
   ```bash
   streamlit run streamlit_app.py
   ```

## Disclaimer
This application is for educational and reference purposes only. Always consult a certified pharmacist or medical professional for pharmaceutical verification.
