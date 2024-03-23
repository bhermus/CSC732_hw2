import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

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

# Split data into 70% training and 30% (20% validation + 10% testing)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Further split the remaining data (30%) into 20% validation and 10% testing
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/3, random_state=42)

# Print the sizes of each subset
print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Testing set size:", len(X_test))

