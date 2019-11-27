import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

#Loading the Boston dataset
from sklearn.datasets import load_boston
boston = load_boston()

#Looking at info about the dataset
print(boston.keys())
print(boston.feature_names)
print(boston.DESCR)

df = pd.DataFrame(boston.data, columns= boston.feature_names)
df['MEDV'] = boston.target


#Keeping only the features that interest the study and dropping the rest. INDUS, RM, DIS, PTRATIO, RAD, MEDV
df.drop(['CRIM', 'ZN', 'CHAS', 'NOX', 'AGE', 'TAX', 'B', 'LSTAT'], axis=1, inplace=True)
# print(df)
print(df.describe())

#Creating different plots

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(df['MEDV'], bins=30)
os.makedirs('figures/', exist_ok=True)
plt.savefig('figures/boston_displot.png', dpi=300)

# # Creating a plot of MEDV and RM
# fig, axes = plt.subplots(1, 1, figsize=(5, 5))
# axes.scatter(df['MEDV'], df['RM'], s=10, label='RM', color='green', marker='^')
# axes.set_title('RM vs MEDV')
# axes.set_xlabel('MEDV')
# axes.set_ylabel('RM')
# axes.legend()
# os.makedirs('plots/', exist_ok=True)
# plt.savefig('plots/boston_room_price_scatter.png', dpi=300)
# plt.close()

#Seeing that we have some outliers in the MDEV, I decided to drop those lines because maybe missing or missing values
print(df[df['MEDV'] == 50.00])
print(df[df['RM'] == 8.78])
indexNames = df[df['MEDV'] == 50.00].index
df.drop(indexNames, inplace=True)
print(df)

#stats

#inflation 394.702% since 1978
df['MEDVUPDATE'] = df['MEDV'] ** 3.94702
# print(df.to_string())

#
# f, ax = plt.subplots(figsize=(12, 12))
# corr = df.select_dtypes(exclude=["object", "bool"]).corr()
#
# # TO display diagonal matrix instead of full matrix.
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
#
# # Generate a custom diverging colormap.
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
# # Draw the heatmap with the mask and correct aspect ratio.
# g = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, annot=True, fmt='.2f', \
#                 square=True, linewidths=.5, cbar_kws={"shrink": .5})
#
# # plt.subplots_adjust(top=0.99)
# plt.title("Diagonal Correlation HeatMap")
# os.makedirs('plots/', exist_ok=True)
# plt.savefig('plots/boston_heatmap.png')
# plt.close()
#

