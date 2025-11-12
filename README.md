# 🎓 Student Performance ML Deployment

This This project was developed as part of the **ANA680: Machine Learning Deployment** midterm.  
The goal is to train a machine learning model that predicts a student's **race/ethnicity** from their **math, reading, and writing scores**, then deploy it to **Heroku** using a **CI/CD pipeline** through **GitHub Actions**.

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
**Example Output:**
Accuracy: 0.23
Model trained and saved as model.pkl

---

📂 File Structure
app.py
train_model.py
model.pkl
requirements.txt
Procfile
templates/index.html
.github/workflows/deploy.yml
