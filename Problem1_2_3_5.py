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

filename = "BankNote_Authentication.csv"
Bank = pd.read_csv(filename)

print('Problem2 Split dataset')

X_data = Bank['variance'].values[:, np.newaxis]
Y_data = Bank['skewness'].values

X_train, X_testVal, Y_train, Y_testVal = train_test_split(X_data, Y_data, test_size=0.30)
X_val, X_test, Y_val, Y_test = train_test_split(X_testVal, Y_testVal, test_size=0.10)


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

print(Y_train.shape)
print(Y_val.shape)
print(Y_test.shape)

print()
print('-----------------------')
print()


print(' Problem3 Scaling data ,  Problem5 R^2 ')

scalers = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler(),
           "Normalizer": Normalizer()}

for scaler_name, scaler in scalers.items():
    # Scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Training
    slr = LinearRegression()
    slr.fit(X_train_scaled, Y_train)

    # Testing
    Y_train_pred = slr.predict(X_train_scaled)
    Y_val_pred = slr.predict(X_val_scaled)
    Y_test_pred = slr.predict(X_test_scaled)

    print(scaler_name, 'results')
    # Print results
    print('MSE train: %.2f, validation: %.2f, test: %.2f' % (
        mean_squared_error(Y_train, Y_train_pred),
        mean_squared_error(Y_val, Y_val_pred),
        mean_squared_error(Y_test, Y_test_pred)
    ))


    print()



    print('R^2 train: %.2f, validation: %.2f, test: %.2f' % (
        r2_score(Y_train, Y_train_pred),
        r2_score(Y_val, Y_val_pred),
        r2_score(Y_test, Y_test_pred)
    ))

   



