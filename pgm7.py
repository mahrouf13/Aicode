from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target
class_names = wine.target_names

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test_scaled)
conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred), index=class_names, columns=class_names)
report = classification_report(y_test, y_pred, target_names=class_names)

# Display results
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)
