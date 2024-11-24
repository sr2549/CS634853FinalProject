import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('diabetes.csv')  # Replace with actual path if needed




# Display first few rows of the dataset
print(data.head())

# 1. **Distribution of Target Variable (Outcome)**

plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=data, palette='Set1')
plt.title('Distribution of Diabetes Outcome (0: No, 1: Yes)')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()

# 2. **Correlation Heatmap**

correlation_matrix = data.drop('Outcome', axis=1).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()

# 3. **Pairplot (Scatterplot Matrix) by Outcome**

sns.pairplot(data, hue='Outcome', diag_kind='kde', palette='Set1')
plt.suptitle('Pairplot of Features by Outcome', y=1.02)
plt.show()

# 4. **Boxplots for Feature Distribution by Outcome**

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='Outcome', y=feature, data=data, palette='Set1')
    plt.title(f'Boxplot of {feature} by Outcome')

plt.tight_layout()
plt.show()

# 5. **Violin Plots for Feature Distribution by Outcome**

plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.violinplot(x='Outcome', y=feature, data=data, palette='Set1')
    plt.title(f'Violin plot of {feature} by Outcome')

plt.tight_layout()
plt.show()

# 6. **Histograms of Features for Both Classes (0 and 1 in Outcome)**

plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.histplot(data=data, x=feature, hue='Outcome', kde=True, palette='Set1', bins=20)
    plt.title(f'Histogram of {feature} by Outcome')

plt.tight_layout()
plt.show()

# 7. **Feature Importance Visualization using Random Forest**

# Train a Random Forest classifier to get feature importances
X = data.drop('Outcome', axis=1)
y = data['Outcome']
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
features = X.columns

# Create a bar plot of feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features, palette='Blues_d')
plt.title('Feature Importance from Random Forest Classifier')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# 8. **Scatter Plot: Age vs Glucose colored by Outcome**

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Glucose', hue='Outcome', data=data, palette='Set1')
plt.title('Age vs Glucose by Outcome')
plt.xlabel('Age')
plt.ylabel('Glucose')
plt.show()

# 9. **Scatter Plot: BMI vs Age colored by Outcome**

plt.figure(figsize=(8, 6))
sns.scatterplot(x='BMI', y='Age', hue='Outcome', data=data, palette='Set1')
plt.title('BMI vs Age by Outcome')
plt.xlabel('BMI')
plt.ylabel('Age')
plt.show()

# 10. **Distribution of Insulin with Log Transformation (excluding zeros)**

plt.figure(figsize=(8, 6))
sns.histplot(data['Insulin'].replace(0, np.nan).dropna(), kde=True, bins=20, color='purple')
plt.title('Distribution of Insulin (excluding 0 values)')
plt.xlabel('Insulin')
plt.ylabel('Frequency')
plt.show()