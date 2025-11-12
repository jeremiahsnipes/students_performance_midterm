# 🎓 Student Performance ML Deployment

This project trains a machine learning model to predict a student's race/ethnicity using their math, reading, and writing scores.  
The model was built in Python, tested locally, and deployed on Heroku through a connected GitHub repo for continuous integration and delivery.

---

## 📘 Dataset
**Source:** [Students Performance in Exams (Kaggle)](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)  
**Variables Used:**
- `math score`
- `reading score`
- `writing score`
- `race/ethnicity` *(target)*

---

## ⚙️ Model Training
The model was trained using a **RandomForestClassifier** from `scikit-learn`.

**Steps:**
1. Load and preprocess the dataset (`StudentsPerformance.csv`).  
2. Split data into train/test sets.  
3. Train and evaluate a RandomForest model.  
4. Save the model as `model.pkl` using `joblib`.

**Script:** `train_model.py`  
**Best Parameters:**  
`{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 300}`  

**Final Accuracy:** `0.345` (34.5%)  
The model achieved modest accuracy due to the weak correlation between exam scores and race/ethnicity categories, but it successfully demonstrates the full ML deployment workflow.

---

📂 File Structure
- app.py
- train_model.py
- model.pkl
- requirements.txt
- Procfile
- templates/
  - index.html
- .github/
  - workflows/
  - deploy.yml

## 🌐 Live App
Heroku: [https://students-performance-midterm-js.herokuapp.com/]([(https://students-performance-midterm-ac846ab83c16.herokuapp.com/))  
GitHub: [https://github.com/jeremiahsnipes/students_performance_midterm](https://github.com/jeremiahsnipes/students_performance_midterm)

---

## 🖥️ Screenshot
**Model running locally**
![Model running locally](Model%20running%20locally.png)
