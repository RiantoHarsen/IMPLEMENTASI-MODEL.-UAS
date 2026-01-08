import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Eksplorasi dataset (EDA singkat)
print("Shape of dataset:", X.shape)
print("Feature names:", feature_names)
print("Target names:", target_names)
print("First 5 rows of features:")
print(pd.DataFrame(X, columns=feature_names).head())
print("Target distribution:")
print(pd.Series(y).value_counts())

# Preprocessing data
# Iris dataset tidak memiliki missing values, tapi kita cek untuk generalisasi
df = pd.DataFrame(X, columns=feature_names)
print("Missing values per column:")
print(df.isnull().sum())

# Encoding: Iris sudah numerical, tidak perlu encoding categorical
# Jika ada categorical, bisa gunakan LabelEncoder atau OneHotEncoder

# Bagi data menjadi training set dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Bangun model Decision Tree
# Parameter penting: max_depth, criterion, dll.
model = DecisionTreeClassifier(
    criterion='gini',  # atau 'entropy'
    max_depth=5,       # batas kedalaman pohon
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualisasi pohon (opsional)
plt.figure(figsize=(20,10))
tree.plot_tree(model, feature_names=feature_names, class_names=target_names, filled=True)
plt.savefig('decision_tree.svg')
print("Decision tree visualization saved as 'decision_tree.svg'")

# Simpan sebagai HTML untuk mudah dibuka di browser
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Decision Tree Visualization</title>
</head>
<body>
    <h1>Decision Tree Model Visualization</h1>
    <p>Accuracy: {accuracy_score(y_test, y_pred):.4f}</p>
    <img src="decision_tree.svg" alt="Decision Tree" style="max-width: 100%; height: auto;">
</body>
</html>
"""

with open('decision_tree_visualization.html', 'w') as f:
    f.write(html_content)
print("HTML visualization saved as 'decision_tree_visualization.html'")
