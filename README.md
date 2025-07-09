# Dynamic 5S Compliance Audits

This is a company-level web application designed to **automatically evaluate 5S compliance** in industrial environments using artificial intelligence. The system detects workplace objects and classifies their condition based on the **5S principles**:  
**Sort (Seiri), Set in Order (Seiton), Shine (Seiso), Standardize (Seiketsu), and Sustain (Shitsuke).**

The goal of this project is to support digital transformation in industrial workplace monitoring through real-time, AI-based 5S audits. This system detects tables and automatically evaluates their condition based on 5S principles.

## Key Features

- Upload **images** or **videos** to evaluate 5S conditions.
- AI-powered analysis using **YOLOv11** and **YOLOv10** for object detection and multi-label classification.
- Front-end web interface with login, profile, and result pages.
- Takes the first frame of uploaded video as a representative frame for evaluation.
- Modular architecture for real-time or batch inference.

## Requirements

- Python 3.8+
- Node.js (for frontend Firebase, optional)
- pip (Python package installer)

## How to Use

### 1. Clone the Repository

  git clone https://github.com/yourusername/dynamic5s.git
  cd dynamic5s

### 2. Install Python Depedencies

  pip install -r requirements.txt

### 3. Run Python App

  python app.py

## Tech Stack

- **YOLOv11 and YOLOv10** – Object detection (Ultralytics) 
- **Python** – Main programming language
- **Google Colab** – Model training/testing
- **Roboflow** – Dataset labelling
- **Pandas / OpenCV / Matplotlib** – Data handling & visualization
