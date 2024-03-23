import matplotlib.pyplot as plt
import pandas as pd

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

# variable information
print(energy_efficiency.variables)


