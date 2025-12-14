# Iris ML Classification Project

## Overview
This project uses **Decision Tree and Random Forest** to classify the **Iris dataset**.  
The dataset contains 150 samples with 4 features each:  
- Sepal length (cm)  
- Sepal width (cm)  
- Petal length (cm)  
- Petal width (cm)  

The goal is to predict the Iris species: **Setosa, Versicolor, or Virginica**.

---

## Features
- Data preprocessing
- Train-test split (80-20)
- Decision Tree classifier
- Random Forest classifier
- Feature importance analysis
- Accuracy, classification report, confusion matrix
- ## Models Used

### 1. Decision Tree
- Trains a single decision tree (max depth=5)  
- Visualizes tree with feature splits  
- Achieves 100% accuracy on test set  
- Confusion matrix shows zero misclassification  

### 2. Random Forest
- Trains 100 decision trees (ensemble)  
- Reduces overfitting and confirms feature importance  
- Accuracy: 100%  
- Most important features: Petal length & Petal width  

---

## How to Run

```bash
python decision_tree.py   # Decision Tree
python random_forest.py   # Random Forest


## How to Run
1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Iris-ML-Project.git
2. cd DecisionTree
3. pip install -r requirements.txt
4.python Decision_tree.py





