from flask import Flask, render_template, request, jsonify
import pickle, time

app = Flask(__name__)
model, scaler = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    data = request.json
    features = [[data['amount'], data['frequency'], data['location_change'], data['device_change']]]
    features = scaler.transform(features)
    prob = model.predict_proba(features)[0][1]
    score = int(prob * 100)
    latency = int((time.time() - start) * 1000)
    return jsonify({'risk_score': score, 'is_fraud': score > 70, 'latency_ms': latency})

if __name__ == '__main__':
    app.run(debug=True)