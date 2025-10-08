# 🚢 Titanic Survival Prediction using Logistic Regression

This project predicts the survival of passengers aboard the Titanic using **Logistic Regression**, a supervised machine learning algorithm.  
It demonstrates a complete ML workflow — from data preprocessing and feature scaling to model evaluation — using Python and Scikit-learn.


## 🧠 Overview
The Titanic dataset is one of the most popular beginner datasets used to learn classification problems.  
The goal is to predict whether a passenger survived (`1`) or not (`0`) based on features like **age**, **gender**, **class**, and **fare**.

---

## 📂 Dataset
The dataset file `Titanic_dataset.csv` should include the following columns:

| Column | Description |
|--------|--------------|
| Survived | Target variable (0 = No, 1 = Yes) |
| Pclass | Passenger class (1st, 2nd, 3rd) |
| Sex | Gender of the passenger |
| Age | Age of the passenger |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Ticket fare |
| Embarked | Port of embarkation (C, Q, S) |
| Cabin, Name, Ticket | Dropped as irrelevant |

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/titanic-logistic-regression.git
   cd titanic-logistic-regression
   ```
## 2.Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## 3.Run the script:
   ```bash
   python main.py
   ```

