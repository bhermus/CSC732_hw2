import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载结果数据集
Bank = pd.read_csv('BankNote_Authentication.csv')



# get column name
X_values = Bank['variance']
Y1_values = Bank['skewness']
Y2_values = Bank['curtosis']
Y3_values = Bank['entropy']
Y4_values = Bank['class']


# use Matplotlib to pain scatter picture
plt.figure(figsize=(10, 6))
plt.scatter(X_values, Y1_values, label='Y1')
plt.scatter(X_values, Y2_values, label='Y2')
plt.scatter(X_values, Y3_values, label='Y3')
plt.scatter(X_values, Y4_values, label='Y4')

plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Scatter Plot of Results')
plt.legend()
plt.grid(True)
plt.show()

# use Seaborn to pain line graph
plt.figure(figsize=(10, 6))
sns.lineplot(x=X_values, y=Y1_values, label='Y1')
sns.lineplot(x=X_values, y=Y2_values, label='Y2')
sns.lineplot(x=X_values, y=Y3_values, label='Y3')
sns.lineplot(x=X_values, y=Y4_values, label='Y4')

plt.xlabel('variance')
plt.ylabel('skewness,curtosis,entropy,class')
plt.title('Line Plot of Results')
plt.legend()
plt.grid(True)
plt.show()
