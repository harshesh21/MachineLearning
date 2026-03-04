import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
from matplotlib.axes import Axes

df = pd.read_csv('Titanic-Dataset.csv')
# print()
# print(f'-----------END OF HEAD -----------')
# print(df.info())
# print(f'-----------END OF INFO -----------')
# print(df.describe())

#df.add('Age_plus_Fare', df['Age'] + df['Fare'], axis=0, inplace=True)
df.drop('PassengerId', axis=1, inplace=True)

imputer = SimpleImputer(strategy='constant')
#print(df)
df2 = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
#print(df2)
#print(df2.info())
df3=df.groupby('Sex')[['Survived','Age']].agg({'Survived': 'sum', 'Age': 'mean'}).reset_index()
print(df3)

#print(df.isnull().sum())

# fig = plt.figure(figsize=(8,6))
# axes = fig.add_subplot(1, 1, 1)
# axes.hist(df['Age'], df['Fare'], bins=30, color='purple', alpha=0.7)
# axes.set_title("Age vs Fare on Titanic")
# axes.set_xlabel("Age")
# axes.set_ylabel("Fare")

# fig.add_subplot(2, 1, 1)
# df['Fare'].hist(bins=30, color='blue', alpha=0.7)
# plt.title("Fare Distribution on Titanic")
# plt.xlabel("Fare")
# plt.ylabel("Number of Passengers")
# plt.grid(alpha=0.5)
x = np.linspace(0, 10, 1000)
y = np.sin(x)

fig, ax = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
axes: list[Axes] = ax.tolist()
axes[0].scatter(x[0:500:10], y[0:500:10], color='red', label='sin(x) points')
axes[0].set_title("Sine Wave")
axes[0].set_xlabel("X axis")
axes[0].set_ylabel("sin(x)")

labels = ['A','B','C']
values = [12, 24, 36]

bars = axes[1].bar(labels, values, color=['red','green','blue'])
bars[0].set_hatch('/')
bars[1].set_hatch('o')  
bars[2].set_hatch('\\')
axes[1].set_title("Bar Chart Example")
axes[1].set_xlabel("Categories")
axes[1].set_ylabel("Values")

data = [np.random.normal(0, std, 100) for std in range(1,4)]
axes[2].boxplot(data, vert=True, patch_artist=True)
axes[2].set_title("Box Plot Example")
axes[2].set_xlabel("Dataset")
axes[2].set_ylabel("Value")

plt.show()
