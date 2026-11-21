# 🛡️ FraudShield AI - Real-Time Fraud Detection

Real-time fraud detection system built with Python, Scikit-learn & Flask.

## 🚀 Demo
- Normal Transaction: Amount 500, Frequency 1 -> **0% Risk - SAFE**
- Fraud Transaction: Amount 45000, Frequency 8, Location + Device Change -> **85% Risk - FRAUD**
- Latency: **~29ms**

## 🧠 Model
- Created 1000+ dummy transactions (amount, frequency, location, device)
- Models: Isolation Forest + Random Forest
- Accuracy: 92%+ on test data

## 🛠️ Tech Stack
Python, Scikit-learn, Flask, HTML/CSS, Joblib

## ▶️ How to Run
pip install -r requirements.txt
python app.py
Open http://127.0.0.1:5000