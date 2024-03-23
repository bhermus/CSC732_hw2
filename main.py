import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

from sklearn.metrics import r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Adjust pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load dataset
X = pd.read_csv('energy_efficiency_features.csv')
y = pd.read_csv('energy_efficiency_targets.csv')

# Set variable information
feature_descriptions = {
    'X1': 'Relative Compactness',
    'X2': 'Surface Area',
    'X3': 'Wall Area',
    'X4': 'Roof Area',
    'X5': 'Overall Height',
    'X6': 'Orientation',
    'X7': 'Glazing Area',
    'X8': 'Glazing Area Distribution'
}
target_variables = {
    'Y1': 'Heating Load',
    'Y2': 'Cooling Load'
}

# Split data into 70% training and 30% (20% validation + 10% testing)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Further split the remaining data (30%) into 20% validation and 10% testing
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/3, random_state=42)

# Print the sizes of each subset
print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Testing set size:", len(X_test))

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and validation sets
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

n_features = X_train_scaled.shape[1]  # 8
weights = np.zeros(n_features)
bias = 0

# Hyperparameters
learning_rate = 0.01
num_epochs = 1000

# GD training loop
for epoch in range(1, num_epochs + 1):
    # Forward pass: compute predictions
    predictions_train = np.dot(X_train_scaled, weights) + bias
    predictions_val = np.dot(X_val_scaled, weights) + bias

    # Compute loss (mean squared error)
    loss_train = np.mean((predictions_train - y_train["Y1"])**2)  # NOTE: We are using Y1, "Heating Load"
    loss_val = np.mean((predictions_val - y_val["Y1"])**2)

    # Backward pass: compute gradients
    grad_weights = np.dot(X_train_scaled.T, predictions_train - y_train["Y1"]) / len(X_train_scaled)
    grad_bias = np.mean(predictions_train - y_train["Y1"])

    # Update model parameters
    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias

    if epoch % 100 == 0:
        print(f"Iteration {epoch}: Training Loss = {loss_train}, Validation Loss = {loss_val}")

# Use the trained weights and bias to make predictions on the validation set
predictions_val = np.dot(X_val_scaled, weights) + bias

# Calculate R-squared
r_squared = r2_score(y_val['Y1'], predictions_val)
print("R-squared:", r_squared)

predictions_test = np.dot(X_test, weights) + bias

# Define a threshold for determining classes
threshold = 10

# Convert continuous predictions to binary classes based on the threshold
predicted_classes = (predictions_test > threshold).astype(int)
actual_classes = (y_test['Y1'] > threshold).astype(int)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(actual_classes, predicted_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
seaborn.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Testing Set)')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.show()
