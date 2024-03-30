import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import figure
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler,Normalizer,RobustScaler
from sklearn.metrics import r2_score,confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle


print(' Problem6. Use regressors demonstrated')


filename = "BankNote_Authentication.csv"
Bank = pd.read_csv(filename)

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    return



# defining independent variables and dependent variable
X_variance = Bank['variance'].values.reshape(-1, 1)  # reshape为2D array
X_skewness = Bank['skewness'].values.reshape(-1, 1)
X_class = Bank['class'].values.reshape(-1, 1)
X_curtosis = Bank['curtosis'].values.reshape(-1, 1)  # correct name
y = Bank['entropy'].values

# GRE scores vs Chance of Admit
slr_variance = LinearRegression()
slr_variance.fit(X_variance, y)
y_variance = slr_variance.predict(X_variance)

plt.figure(figsize=(13, 16))
plt.subplot(3, 2, 1)
lin_regplot(X_variance, y, slr_variance)
plt.title('Slope (w1) = %.2f\nIntercept/bias (w0): %.2f' % (slr_variance.coef_[0], slr_variance.intercept_))  # change title
plt.xlabel('variance')
plt.ylabel('entropy')
plt.tight_layout()  # adjust layout
plt.show()
