# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load data
df = pd.read_csv('StudentsPerformance.csv')

# 2. Feature selection (add context)
X = df[['math score', 'reading score', 'writing score',
        'gender', 'lunch', 'test preparation course', 'parental level of education']]
y = df['race/ethnicity']

# 3. Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Tune Random Forest (quick grid)
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# 6. Evaluate
pred = best_model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# 7. Save trained model
# 7. Save trained model and columns
joblib.dump(best_model, 'model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')
print("\nModel and training columns saved!")

