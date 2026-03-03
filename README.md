
# 😷 Face Mask Detection API

A Deep Learning based Face Mask Detection API built using **TensorFlow** and **FastAPI**.

This project detects whether a person is wearing a mask or not from an uploaded image.

---

## 🚀 Tech Stack

- Python
- TensorFlow / Keras
- FastAPI
- OpenCV
- Uvicorn

---

## 📂 Project Structure

face-mask-detection-api/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

---

## ⚙️ Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/face-mask-detection-api.git

Go inside the folder:

cd face-mask-detection-api

Create virtual environment:

python -m venv venv

Activate venv (Windows):

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

---

## ▶️ Run the API

uvicorn app:app --reload

Open in browser:

http://127.0.0.1:8000/docs

---

## 📸 API Endpoint

POST /predict/

Upload an image and get prediction:

Response Example:

{
  "prediction": "With Mask"
}

---

## 🎯 Model Accuracy

- Training Accuracy: 97%
- Validation Accuracy: 93%

---

## 📌 Author

Your Name
