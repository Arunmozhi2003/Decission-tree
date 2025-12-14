import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier as decisiontree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix



# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
print(X.head())
print(y.value_counts())
print(X.info())

min_values = X.min()
max_values=X.max()
print("Maximum values of each feature:\n", max_values)
print("Minimum values of each feature:\n", min_values)


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
print("Training feature set shape:", X_train.shape)
print("Testing feature set shape:", X_test.shape)
print("Training labels shape:", y_train.shape)  
print("Testing labels shape:", y_test.shape)
print(X_train.head())
print(y_train.head())

# Initialize and train the model
model = decisiontree(max_depth=5, random_state=42)

model.fit(X_train, y_train)
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree – Iris Classification")
plt.show()

# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
#we upgrade to confusion matrix
print(confusion_matrix(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)


print(X_train,X_test,y_train,y_test, sep="\n")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  
importances = model.feature_importances_

print("Feature Importances:")
for name, val in zip(X.columns, importances):
    print(f"{name}: {val:.3f}")


print("Classification Report for Random Forest:\n", classification_report(y_test, y_pred))
#we upgrade to confusion matrix
