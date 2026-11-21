import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
normal = pd.DataFrame({
    'amount': np.random.randint(100, 30000, 850),
    'frequency_last_hour': np.random.randint(1, 5, 850),
    'location_change': np.random.choice([0,1], 850, p=[0.8, 0.2]),
    'device_change': np.random.choice([0,1], 850, p=[0.9, 0.1]),
})
normal['is_fraud'] = 0
fraud = pd.DataFrame({
    'amount': np.random.randint(20000, 50000, 150),
    'frequency_last_hour': np.random.randint(3, 10, 150),
    'location_change': np.random.choice([0,1], 150, p=[0.2, 0.8]),
    'device_change': np.random.choice([0,1], 150, p=[0.2, 0.8]),
})
fraud['is_fraud'] = 1
df = pd.concat([normal, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
noise_idx = np.random.choice(df.index, size=10, replace=False)
df.loc[noise_idx, 'is_fraud'] = 1 - df.loc[noise_idx, 'is_fraud']
X = df[['amount','frequency_last_hour','location_change','device_change']]
y = df['is_fraud']

for seed in range(0, 50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"Seed {seed} -> Accuracy {acc:.3f}")