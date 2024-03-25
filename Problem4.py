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

print('Split dataset')

X_data = Bank['variance'].values[:, np.newaxis]
Y_data = Bank['skewness'].values


w = 1.0

def forward(x):
    return x * w


def loss(xs, ys):
    y_pred = forward(x)
    return (y_pred - ys) ** 2


def gradient(xs, ys):
    return 2 * xs * (xs * w - ys)


print('Predict (before training)', 4, forward(4))

epoch_list = []
loss_list = []

for epoch in range(100):
    for x, y in zip(X_data, Y_data):
        grad_val = gradient(x, y)
        w -= 0.01 * grad_val
        print('\tgrad:', x, y, grad_val)
        l = loss(x, y)

    print('progress:', epoch, 'w=', w, 'loss=', l)
    epoch_list.append(epoch)
    loss_list.append(l)

print('Predict (after training)', 4, forward(4))

plt.plot(epoch_list, loss_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()
