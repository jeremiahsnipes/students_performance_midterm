# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv('StudentsPerformance.csv')

# Features & target
X = df[['math score', 'reading score', 'writing score']]
y = df['race/ethnicity']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, 'model.pkl')
