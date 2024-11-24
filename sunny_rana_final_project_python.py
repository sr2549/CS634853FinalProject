# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
# Load and preprocess the dataset
data = pd.read_csv('diabetes.csv')  # Replace with actual path if needed
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Target variable


# Encode target if it's not numerical (e.g., 0 or 1)
y = pd.factorize(y)[0]

# Define 10-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize structures to collect metrics
metrics_dict = {"RandomForest": [], "SVM": [], "LSTM": []}

# Function to calculate metrics based on TP, TN, FP, FN, and predictions
def calculate_metrics(TP, TN, FP, FN, y_test, y_pred, y_pred_probs=None):
    # Calculate basic metrics
    TPR = TP / (TP + FN) if TP + FN > 0 else 0  # True Positive Rate (Recall)
    TNR = TN / (TN + FP) if TN + FP > 0 else 0  # True Negative Rate
    FPR = FP / (FP + TN) if FP + TN > 0 else 0  # False Positive Rate
    FNR = FN / (FN + TP) if FN + TP > 0 else 0  # False Negative Rate
    Precision = TP / (TP + FP) if TP + FP > 0 else 0  # Precision
    F1_measure = 2 * (Precision * TPR) / (Precision + TPR) if Precision + TPR > 0 else 0  # F1 Score
    Accuracy = (TP + TN) / (TP + TN + FP + FN)  # Accuracy
    Error_rate = (FP + FN) / (TP + TN + FP + FN)  # Error Rate
    BACC = (TPR + TNR) / 2  # Balanced Accuracy
    TSS = TPR - FPR  # True Skill Statistic
    HSS = (2 * (TP * TN - FP * FN)) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))  # Heidke Skill Score

    # Brier Score and AUC require predicted probabilities
    if y_pred_probs is not None:
        Brier_score = brier_score_loss(y_test, y_pred_probs)  # Brier Score
        AUC = roc_auc_score(y_test, y_pred_probs)  # AUC (Area Under Curve)
    else:
        Brier_score = None
        AUC = None

    return TPR, TNR, FPR, FNR, Precision, F1_measure, Accuracy, Error_rate, BACC, TSS, HSS, Brier_score, AUC

# Model 1: Random Forest Classifier
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_probs = rf.predict_proba(X_test)[:, 1]  # Probability of the positive class
    
    # Confusion matrix
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate and store metrics
    TPR, TNR, FPR, FNR, Precision, F1_measure, Accuracy, Error_rate, BACC, TSS, HSS, Brier_score, AUC = calculate_metrics(
        TP, TN, FP, FN, y_test, y_pred, y_pred_probs)
    
    metrics_dict["RandomForest"].append([TP, TN, FP, FN, TPR, TNR, FPR, FNR, Precision, F1_measure, Accuracy, Error_rate, BACC, TSS, HSS, Brier_score, AUC])

# Model 2: Support Vector Machine (SVM)
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    svm = SVC(kernel='linear', probability=True, random_state=42)  # Enable probability estimation
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_pred_probs = svm.predict_proba(X_test)[:, 1]  # Probability of the positive class
    
    # Confusion matrix
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate and store metrics
    TPR, TNR, FPR, FNR, Precision, F1_measure, Accuracy, Error_rate, BACC, TSS, HSS, Brier_score, AUC = calculate_metrics(
        TP, TN, FP, FN, y_test, y_pred, y_pred_probs)
    
    metrics_dict["SVM"].append([TP, TN, FP, FN, TPR, TNR, FPR, FNR, Precision, F1_measure, Accuracy, Error_rate, BACC, TSS, HSS, Brier_score, AUC])

# Model 3: LSTM (Deep Learning)
X_lstm = np.expand_dims(X.values, axis=-1)  # Expand dimensions for LSTM input
y_lstm = to_categorical(y)  # One-hot encode labels for neural network

# Define the LSTM model **outside** the loop to avoid retracing
lstm_model = Sequential([
    Input(shape=(X_lstm.shape[1], 1)),  # Using Input layer for shape
    LSTM(50),
    Dense(2, activation='softmax')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Cross-validation loop for LSTM
for train_index, test_index in kf.split(X_lstm, np.argmax(y_lstm, axis=1)):
    X_train, X_test = X_lstm[train_index], X_lstm[test_index]
    y_train, y_test = y_lstm[train_index], y_lstm[test_index]
    
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    y_pred = np.argmax(lstm_model.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)
    
    # Get predicted probabilities for AUC and Brier score
    y_pred_probs = lstm_model.predict(X_test)[:, 1]  # Assuming binary classification
    
    # Confusion matrix
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate and store metrics
    TPR, TNR, FPR, FNR, Precision, F1_measure, Accuracy, Error_rate, BACC, TSS, HSS, Brier_score, AUC = calculate_metrics(
        TP, TN, FP, FN, y_test, y_pred, y_pred_probs)
    
    metrics_dict["LSTM"].append([TP, TN, FP, FN, TPR, TNR, FPR, FNR, Precision, F1_measure, Accuracy, Error_rate, BACC, TSS, HSS, Brier_score, AUC])

# Convert metrics to DataFrame for display
metrics_df = {model: pd.DataFrame(values, columns=['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR', 
                                                   'Precision', 'F1_measure', 'Accuracy', 'Error_rate', 
                                                   'BACC', 'TSS', 'HSS', 'Brier_score', 'AUC'])
              for model, values in metrics_dict.items()}

# Display results for each model
for model, df in metrics_df.items():
    print(f"\nResults for {model}:")
    print(df)
    print("Average metrics across 10 folds:")
    print(df.mean())